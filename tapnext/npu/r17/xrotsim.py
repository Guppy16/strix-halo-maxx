"""Round 17 (X1 full rotation, mod 4): region-level model of the SSM area over frames, from the real plans of the 4 phases.

Memory = {region offset in S (column 0): logical value}. Per frame t (phase r = t mod 4): the yx GEMM writes ("y", t) at its
drain base and ("x", t) at its second half (x_at); the SSM fill (head 0, column 0) reads 7 streams [y, R0..R3, Hlo, Hhi] and the
kernel maps tap x - k to R[(r - k) mod 4] (ew8.cc -DXROT): they must be y(t), x(t), x(t-1), x(t-2), x(t-3), hlo(t-1), hhi(t-1)
(frame 0 after reset: zeros); the drain writes h lo, h hi, z; the out GEMM must read z(t).

    python3 r17/xrotsim.py
"""

import plan8


def main():
    plan8.XROT, plan8.P23, plan8.A_HALF, plan8.P60G = 4, True, True, True
    S = plan8.S
    plans = [plan8.full_step(1, phase=k) for k in range(4)]
    mem = {}
    for t in range(14):
        r = t % 4
        p, info = plans[r]
        a = info["areas"][0]
        rel = lambda off: (off - a["base"]) / S  # noqa: E731
        ops = {op["name"].split("_", 1)[1]: op for op in p.ops if op["name"].startswith("b0_")}
        d = ops["yx"]["c"][0]
        assert d["sizes"] == [2, 12, plan8.RB, plan8.XB]
        mem[rel(d["off"])], mem[rel(d["off"] + d["strides"][0])] = ("y", t), ("x", t)
        fill = ops["ssm"]["fills"][0][0][1]
        assert fill["sizes"] == [plan8.RB, 7, 1, plan8.XB] and fill["strides"][1] == S
        parts = [mem.get(rel(fill["off"]) + j, "zero") for j in range(7)]
        R = parts[1:5]
        got = [parts[0]] + [R[(r - k) % 4] for k in range(4)] + parts[5:]
        z = lambda v, tt: (v, tt) if tt >= 0 else "zero"  # noqa: E731
        want = [("y", t), ("x", t), z("x", t - 1), z("x", t - 2), z("x", t - 3), z("hlo", t - 1), z("hhi", t - 1)]
        assert got == want, (t, got, want)
        dr = ops["ssm"]["c"][0]
        assert dr["sizes"][2] == 3
        for j, v in enumerate([("hlo", t), ("hhi", t), ("z", t)]):
            mem[rel(dr["off"]) + j] = v
        assert mem[rel(ops["out"]["src"]["base"])] == ("z", t)
        print(f"frame {t} phase {r}: x -> R{r}, taps OK")


if __name__ == "__main__":
    main()
