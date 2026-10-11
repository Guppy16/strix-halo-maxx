"""Round 21 E7'' (research pass 17, B on both FIFOs): split every GEMM core's B object over two L1 buffers in the LOWERED MLIR
(after r19/csplit.py, before aiecc): object bytes [0, YB) -> the IRON B buffer (Y, bank 3), [YB, OBJ) -> a new X buffer (bank 0).
Per GEMM tile and B buffer i the S2MM BD [acq prod; buff_i; rel cons] becomes
    [acq prod 1; Y_i words 0 .. YB/4; rel e7h 1]  ->  [acq e7h 1; X_i 0 .. (OBJ - YB)/4; rel cons 1]
(a BD block needs one acquire AND one release; an acquire by 0 is refused at CDO generation: a hand-off lock e7h (init 0) per tile). The X buffers sit at fixed addresses (default: the top
1,152 B of a stack raised by design8 E7=1: 0x70A00 / 0x70C40), reported for mm8.cc's KS_X0 / KS_X1.
usage: python3 r21/e7patch.py IN.mlir OUT.mlir [--yb 4032] [--obj 4608] [--x 2560,3136]"""
import argparse
import re

ap = argparse.ArgumentParser()
ap.add_argument("inp")
ap.add_argument("out")
ap.add_argument("--yb", type=int, default=4032)
ap.add_argument("--lock", type=int, default=7, help="the hand-off lock id (free on every GEMM tile)")
ap.add_argument("--obj", type=int, default=4608)
ap.add_argument("--x", default="2560,3136", help="X buffer addresses (bytes in the tile's 64 KB)")
ap.add_argument("--core-data", type=int, default=3712, help="the GEMM cores' core_data start (after the X buffers); (design8 E7=1 raises the"
                " stack by the X buffers; the patch lowers it back)")
o = ap.parse_args()
xs = [int(v) for v in o.x.split(",")]
xw, yw = (o.obj - o.yb) // 4, o.yb // 4
assert o.yb % 64 == 0 and len(xs) == 2
lines = open(o.inp).read().split("\n")
src = "\n".join(lines)
tiles = sorted(set(re.findall(r"%B(\d)L1_(\d)_cons_buff_0 = aie\.buffer\(%tile_(\d)_(\d)\)", src)))
assert len(tiles) == 24, len(tiles)
n = 0
for r, c, tc, tr in tiles:
    tile = f"tile_{tc}_{tr}"
    pre = f"B{r}L1_{c}_cons"
    k = next(j for j, ln in enumerate(lines) if f"%{pre}_buff_0 = aie.buffer(%{tile})" in ln)
    for i, a in enumerate(xs):
        lines.insert(k, f'    %bx{tc}{tr}_{i} = aie.buffer(%{tile}) {{address = {a} : i32, mem_bank = 0 : i32, sym_name = "bx{tc}{tr}_{i}"}} : memref<{xw}xui32>')
    m0 = next(j for j, ln in enumerate(lines) if re.search(rf"= aie\.mem\(%{tile}\) \{{", ln))
    m1 = next(j for j in range(m0 + 1, len(lines)) if lines[j].startswith("    }"))
    used = {int(v) for ln in lines[m0:m1] for v in re.findall(r"bd_id = (\d+)", ln)}
    assert not re.search(rf"aie\.lock\(%{tile}, {o.lock}\)", "\n".join(lines)), (tile, o.lock)
    lines.insert(k, f'    %e7h{tc}{tr} = aie.lock(%{tile}, {o.lock}) {{init = 0 : i32, sym_name = "e7h{tc}{tr}"}}')
    m0 += 1
    m1 = next(j for j in range(m0 + 1, len(lines)) if lines[j].startswith("    }"))
    for i in range(2):
        j = next(j for j in range(m0, m1) if re.search(rf"aie\.dma_bd\(%{pre}_buff_{i} : memref<(\d+)xui32> offset = 0 len = \1\)", lines[j]))
        acq, bd, rel, nxt = lines[j - 1], lines[j], lines[j + 1], lines[j + 2]
        assert f"%{pre}_prod_lock_0" in acq and "AcquireGreaterEqual" in acq, acq
        assert f"%{pre}_cons_lock_0" in rel and "Release" in rel, rel
        bid, nid = map(int, re.search(r"\{bd_id = (\d+) : i32, next_bd_id = (\d+) : i32\}", bd).groups())
        xid = next(b for b in range(16) if b not in used)
        used.add(xid)
        lab = f"^e7x{tc}{tr}_{i}"
        loc = re.search(r"( loc\(.*\))$", bd)
        loc = loc.group(1) if loc else ""
        ind = bd[:len(bd) - len(bd.lstrip())]
        bdy = re.sub(r"offset = 0 len = \d+\) \{bd_id = (\d+) : i32, next_bd_id = \d+ : i32\}",
                     rf"offset = 0 len = {yw}) {{bd_id = \1 : i32, next_bd_id = {xid} : i32}}", bd)
        mt = re.search(r"memref<(\d+)xui32>", bd).group(1)
        assert int(mt) * 4 >= o.yb, (mt, o.yb)
        new = [acq, bdy, rel.replace(f"%{pre}_cons_lock_0", f"%e7h{tc}{tr}"), f"{ind}aie.next_bd {lab}{loc}",
               f"  {lab}:",
               acq.replace(f"%{pre}_prod_lock_0", f"%e7h{tc}{tr}"),
               f"{ind}aie.dma_bd(%bx{tc}{tr}_{i} : memref<{xw}xui32> offset = 0 len = {xw}) {{bd_id = {xid} : i32, next_bd_id = {nid} : i32}}{loc}",
               rel, nxt]
        lines[j - 1:j + 3] = new
        m1 += 5
        n += 1
# design8 E7=1 raised the GEMM cores' stack by the X buffers' 1,152 B, so core_data already starts behind them (the elfs keep the
# KSC addresses); here the stack goes back to its old size and the X buffers take the freed top
nd = 0
for j, ln in enumerate(lines):
    m = re.search(r"%core_data_(\d)_(\d) = aie\.buffer\(%tile_\d_\d\) \{address = (\d+) : i32", ln)
    if m and int(m.group(2)) in (2, 3, 4):
        assert int(m.group(3)) == o.core_data, (ln[:100], o.core_data)
        tile = f"tile_{m.group(1)}_{m.group(2)}"
        k0 = next(i for i, l2 in enumerate(lines) if re.search(rf"= aie\.core\(%{tile}\) \{{", l2))
        k1 = next(i for i in range(k0 + 1, len(lines)) if lines[i].startswith("    }"))
        assert f"stack_size = {o.core_data} : i32" in lines[k1], lines[k1][-80:]
        lines[k1] = lines[k1].replace(f"stack_size = {o.core_data} : i32", f"stack_size = {min(xs)} : i32")
        nd += 1
assert nd == 24, nd
open(o.out, "w").write("\n".join(lines))
print(f"e7patch: {n} B BDs split at {o.yb} B (X {xw * 4} B at {xs}) on {len(tiles)} tiles")
