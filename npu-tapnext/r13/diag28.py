import sys; sys.path.insert(0, "../r11"); sys.path.insert(0, "../r12"); sys.path.insert(0, ".")
import gen_aexp as G, cpsched
II = int(sys.argv[1])
for name, kw in (("x12 dm5 no maxlife", dict(budgets={"x": 12, "dm": 5})),
                 ("x12 dm5 maxlife", dict(budgets={"x": 12, "dm": 5}, max_life=2 * II)),
                 ("x8 dm3 maxlife", dict(budgets={"x": 8, "dm": 3}, max_life=2 * II)),
                 ("x6 dm3 maxlife3", dict(budgets={"x": 6, "dm": 3}, max_life=3 * II))):
    lp = G.mk()
    res = cpsched.solve(lp, II, maxdist=G.maxdist(II), last_stage=["PU", "FL"], time_limit=60, **kw)
    print(II, name, None if res is None else res[1], flush=True)
