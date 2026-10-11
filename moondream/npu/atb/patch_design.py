"""Generalise the ATB config1 design + kernel: m (64/128), n (multiple of 16), mtk (multiple of 64 dividing K)."""

d = open("enc_gemm.py").read()
d = d.replace(
    '_AIE_KERNELS_INC = Path(__file__).resolve().parents[5] / "aie_kernels"',
    '_AIE_KERNELS_INC = Path("/work/mlir-aie/aie_kernels")',
)
d = d.replace("    n: CompileTime[int] = 128,\n):", "    n: CompileTime[int] = 128,\n    mtk: CompileTime[int] = 512,\n):")
d = d.replace("    mtk = 512\n", "")
d = d.replace('kernel_flags = [f"-I{_AIE_KERNELS_INC}"]', 'kernel_flags = [f"-I{_AIE_KERNELS_INC}", f"-DDIM_M={m}", f"-DDIM_N={n}"]')
# drain every task group in issue order, keeping at most 4 in flight (works for any number of groups)
a = d.index("    def sequence(a, b, c, A_prods, B_prods, C_conses):")
b = d.index("    rt = Runtime(")
d = (
    d[:a]
    + '''    def sequence(a, b, c, A_prods, B_prods, C_conses):
        pending: list[TaskGroup] = []
        for group_idx in range(num_groups):
            tg = TaskGroup()
            a_base_idx = (group_idx // num_col_tile) * n_aie_rows
            for row in range(n_aie_rows):
                A_prods[row].fill(a, tap=A_taps[a_base_idx + row], group=tg, wait=False)
            b_base_idx = (group_idx % num_col_tile) * n_aie_cols
            for col in range(n_aie_cols):
                B_prods[col].fill(b, tap=B_taps[b_base_idx + col], group=tg, wait=False)
            for col in range(n_aie_cols):
                C_conses[col].drain(
                    c, tap=C_taps[group_idx // num_col_tile, b_base_idx + col], group=tg, wait=True
                )
            pending.append(tg)
            if len(pending) == tb_max_n_rows:
                pending.pop(0).finish()
                pending.pop(0).finish()
        for tg in pending:
            tg.finish()

'''
    + d[b:]
)
d = d.replace('    p.add_argument("-n", type=int, default=128)\n', '    p.add_argument("-n", type=int, default=128)\n    p.add_argument("--mtk", type=int, default=512)\n')
d = d.replace("m=opts.m, k=opts.k, n=opts.n)", "m=opts.m, k=opts.k, n=opts.n, mtk=opts.mtk)")
assert "mtk=opts.mtk" in d and "DIM_M" in d and "pending.pop(0)" in d
open("enc_gemm.py", "w").write(d)

k = open("mm_bfp_mixed.cc").read()
k = k.replace("constexpr int m = 128;", "constexpr int m = DIM_M;")
k = k.replace("constexpr int n = 128;", "constexpr int n = DIM_N;")
k = k.replace(
    "static_assert(m * rho == 512 && k == 64 && n == 128 && rho == 4);",
    "static_assert(k == 64 && rho == 4 && (m / rho) % 16 == 0 && n % 16 == 0);",
)
assert "DIM_M" in k and "DIM_N" in k and "(m / rho) % 16" in k
open("mm_bfp_mixed.cc", "w").write(k)
print("patched")
