"""Host <-> NPU transfer costs (median / p99 over many runs). Needs the xfer_<n>.xclbin builds.

    PYTHONPATH=/opt/xilinx/xrt/python python3 xfer_run.py [--iters 300]

Prints:
  UPLOAD   bo.write + sync TO_DEVICE per size and bo flag (host_only / cacheable), and mapped writes (np view) + sync
  LAUNCH   run() -> wait() of the 1-object program (4 KB in, 2 KB out) = dispatch floor, and per frame-size program
  READBACK sync FROM_DEVICE + bo.read of a 576 B result (48 points x (y, x, vis) fp32) and of 2 KB
  E2E      frame write+sync -> dispatch -> result sync+read, for the uint8 and bf16 256 px frames
"""

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/opt/xilinx/xrt/python")
import pyxrt  # noqa: E402

ROOT = "/work/npu-tapnext"
TO, FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
FLAGS = {"host_only": pyxrt.bo.host_only, "cacheable": pyxrt.bo.cacheable}


def stats(ts):
    a = np.array(ts) * 1e6
    return f"median {np.median(a):7.1f} us  p99 {np.percentile(a, 99):7.1f}  min {a.min():7.1f}"


def gpu_busy():
    try:
        return int(open("/sys/class/drm/card1/device/gpu_busy_percent").read())
    except OSError:
        return -1


class Prog:
    def __init__(self, dev, name):
        b = f"{ROOT}/build/{name}"
        self.cfg = json.load(open(b + ".json"))
        self.xb = pyxrt.xclbin(b + ".xclbin")
        dev.register_xclbin(self.xb)
        self.ctx = pyxrt.hw_context(dev, self.xb.get_uuid())
        self.kern = pyxrt.kernel(self.ctx, self.xb.get_kernels()[0].get_name())
        instr = np.fromfile(b + ".bin", dtype=np.uint32)
        self.bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, self.kern.group_id(1))
        self.bo_i.write(instr.tobytes(), 0)
        self.bo_i.sync(TO)
        self.n_instr = len(instr)
        self.in_bytes = self.cfg["COLS"] * self.cfg["n_obj"] * self.cfg["OBJ"] * 4
        self.out_bytes = self.cfg["COLS"] * self.cfg["OUT"] * 4

    def run(self, bo_x, bo_y, timeout_us=3_000_000):
        r = self.kern(3, self.bo_i, self.n_instr, bo_x, bo_y)
        st = r.wait(timeout_us)
        assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st


def expected(x_words, cfg):
    cols, n_obj, obj, out = cfg["COLS"], cfg["n_obj"], cfg["OBJ"], cfg["OUT"]
    v = x_words.reshape(cols, n_obj * obj // out, out)
    return np.bitwise_xor.reduce(v, axis=1).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    o = ap.parse_args()
    dev = pyxrt.device(0)
    print(f"gpu_busy% {gpu_busy()}")
    sizes = {"256u8 (192 KB)": 6, "256bf16 (384 KB)": 12, "720p u8 (2.7 MB)": 85, "1080p u8 (6.2 MB)": 190}
    progs = {name: Prog(dev, f"xfer_{n}") for name, n in sizes.items()}
    p1 = Prog(dev, "xfer_1")

    # --- UPLOAD: write + sync per flag, and mapped writes --------------------------------------------------
    print("\nUPLOAD (host -> device), per call:")
    for name, prog in progs.items():
        nbytes = prog.in_bytes
        src = np.random.randint(0, 2**32, nbytes // 4, dtype=np.uint32)
        for fname, flag in FLAGS.items():
            bo = pyxrt.bo(dev, nbytes, flag, prog.kern.group_id(3))
            tw, ts = [], []
            for _ in range(o.iters):
                t0 = time.perf_counter()
                bo.write(src.tobytes(), 0)
                t1 = time.perf_counter()
                bo.sync(TO)
                t2 = time.perf_counter()
                tw.append(t1 - t0)
                ts.append(t2 - t1)
            print(f"  {name:18s} {fname:9s} write {stats(tw)} | sync {stats(ts)}")
            # mapped: numpy view over bo.map(), copy in with np.copyto (no tobytes), then sync
            try:
                view = np.frombuffer(bo.map(), dtype=np.uint32)
                tm, ts2 = [], []
                for _ in range(o.iters):
                    t0 = time.perf_counter()
                    np.copyto(view, src)
                    t1 = time.perf_counter()
                    bo.sync(TO)
                    t2 = time.perf_counter()
                    tm.append(t1 - t0)
                    ts2.append(t2 - t1)
                print(f"  {name:18s} {fname:9s} mapped-copy {stats(tm)} | sync {stats(ts2)}")
            except Exception as e:  # noqa: BLE001
                print(f"  {name:18s} {fname:9s} map failed: {e}")
            del bo

    # --- LAUNCH: dispatch floor, and per frame size (includes the shim DMA of the frame) -------------------
    print("\nLAUNCH (run() -> wait() returns), host_only bos, after 30 warm-up dispatches:")
    for name, prog in [("1 obj (4 KB in, 2 KB out)", p1)] + list(progs.items()):
        src = np.random.randint(0, 2**32, prog.in_bytes // 4, dtype=np.uint32)
        bo_x = pyxrt.bo(dev, prog.in_bytes, pyxrt.bo.host_only, prog.kern.group_id(3))
        bo_y = pyxrt.bo(dev, prog.out_bytes, pyxrt.bo.host_only, prog.kern.group_id(4))
        bo_x.write(src.tobytes(), 0)
        bo_x.sync(TO)
        prog.run(bo_x, bo_y)
        bo_y.sync(FROM)
        got = np.frombuffer(bo_y.read(prog.out_bytes, 0), dtype=np.uint32)
        ok = np.array_equal(got, expected(src, prog.cfg))
        tl = []
        for it in range(30 + o.iters):
            t0 = time.perf_counter()
            prog.run(bo_x, bo_y)
            t1 = time.perf_counter()
            if it >= 30:
                tl.append(t1 - t0)
        print(f"  {name:28s} {stats(tl)}  check {'OK' if ok else 'MISMATCH'}")

    # --- READBACK -------------------------------------------------------------------------------------------
    print("\nREADBACK (sync FROM_DEVICE + bo.read):")
    for nbytes, label in ((576, "48 points x (y, x, vis) fp32 = 576 B"), (2048, "2 KB"), (1024 * 768 * 2, "1.5 MB (a [1024, 768] bf16 activation)")):
        for fname, flag in FLAGS.items():
            bo = pyxrt.bo(dev, nbytes, flag, p1.kern.group_id(4))
            ts, tr = [], []
            for _ in range(o.iters):
                t0 = time.perf_counter()
                bo.sync(FROM)
                t1 = time.perf_counter()
                bo.read(nbytes, 0)
                t2 = time.perf_counter()
                ts.append(t1 - t0)
                tr.append(t2 - t1)
            print(f"  {label:40s} {fname:9s} sync {stats(ts)} | read {stats(tr)}")

    # --- E2E: frame in host RAM -> dispatch -> points in host RAM ------------------------------------------
    print("\nE2E (write+sync frame -> run -> sync+read 576 B), host_only:")
    for name in ("256u8 (192 KB)", "256bf16 (384 KB)"):
        prog = progs[name]
        src = np.random.randint(0, 2**32, prog.in_bytes // 4, dtype=np.uint32)
        bo_x = pyxrt.bo(dev, prog.in_bytes, pyxrt.bo.host_only, prog.kern.group_id(3))
        bo_y = pyxrt.bo(dev, prog.out_bytes, pyxrt.bo.host_only, prog.kern.group_id(4))
        view = np.frombuffer(bo_x.map(), dtype=np.uint32)
        te, tk = [], []
        for it in range(30 + o.iters):
            t0 = time.perf_counter()
            np.copyto(view, src)
            bo_x.sync(TO)
            t1 = time.perf_counter()
            prog.run(bo_x, bo_y)
            t2 = time.perf_counter()
            bo_y.sync(FROM)
            bo_y.read(576, 0)
            t3 = time.perf_counter()
            if it >= 30:
                te.append(t3 - t0)
                tk.append(t2 - t1)
        print(f"  {name:18s} total {stats(te)} | of which run {stats(tk)}")
    print(f"gpu_busy% {gpu_busy()}")


if __name__ == "__main__":
    main()
