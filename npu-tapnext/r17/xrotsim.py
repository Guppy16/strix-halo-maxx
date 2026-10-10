"""Round 17 (X1 full rotation): region-level model of the SSM area over frames, from the real plans of every phase.

Memory = {region offset (multiple of S inside column 0's span): logical value}. Per frame t (phase t mod N): the yx GEMM writes
("y", t) / ("x", t) at its dst, the SSM reads its 7 streams (fill TAP of head 0, column 0) and must see y(t), x(t), x(t-1), x(t-2),
x(t-3), hlo(t-1), hhi(t-1) (frame 0 after reset: zeros), then writes its drain streams (wrap: x(t), x(t-1), x(t-2), h; rotating:
h, z), and the out GEMM must read z(t). Reads of a stream the same chunk's drain overwrites are fine (the chunk reads first).

    python3 r17/xrotsim.py [N]
"""

import sys

import plan8


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    plan8.XROT = N
    S = plan8.S
    plans = [plan8.full_step(1, phase=k) for k in range(N)]
    mem = {}
    for t in range(3 * N + 2):
        p, info = plans[t % N]
        a = info["areas"][0]
        rel = lambda off: (off - a["area"]) / S  # noqa: E731
        ops = {op["name"].split("_", 1)[1]: op for op in p.ops if op["name"].startswith("b0_")}
        yx = ops["yx"]["dst"]
        mem[rel(yx["base"])], mem[rel(yx["base"]) + 1] = ("y", t), ("x", t)
        ssm = ops["ssm"]
        fill = ssm["fills"][0][0][1]
        assert fill["sizes"] == [plan8.RB, 7, 1, plan8.XB] and fill["strides"][1] == S
        got = [mem.get(rel(fill["off"]) + j, "zero") for j in range(7)]
        z = lambda v, tt: (v, tt) if tt >= 0 else "zero"  # noqa: E731
        want = [("y", t), ("x", t), z("x", t - 1), z("x", t - 2), z("x", t - 3), z("hlo", t - 1), z("hhi", t - 1)]
        assert got == want, (t, got, want)
        dr = ssm["c"][0]
        vals = [("x", t), ("x", t - 1), ("x", t - 2), ("hlo", t), ("hhi", t), ("z", t)] if dr["sizes"][2] == 6 else [("hlo", t), ("hhi", t), ("z", t)]
        for j, v in enumerate(vals):
            mem[rel(dr["off"]) + j] = v if v[1] >= 0 else "zero"
        assert mem[rel(ops["out"]["src"]["base"])] == ("z", t)
        assert 0 <= rel(fill["off"]) and rel(dr["off"]) + len(vals) <= N + 7
        print(f"frame {t} phase {t % N}: reads at {rel(fill['off']):.0f} S, writes {len(vals)} streams at {rel(dr['off']):.0f} S: OK")


if __name__ == "__main__":
    main()
