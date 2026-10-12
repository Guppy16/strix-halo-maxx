"""Round 25 RRES edits (plan8 / pack8 / step8 / design8 / ew8): the 3-input norms' residual (stream 2) for row blocks < RRB parked in
memtile half 1 during the producer GEMM and fed to the EW core's S2MM 1 (a 2 x 1 KB ring) instead of the DDR input ring."""
import os
from pathlib import Path

D = Path(os.environ.get("SHM_SCRATCH") or Path(__file__).resolve().parents[4] / ".scratch") / "npu-tapnext"


def rep(f, pairs):
    p = D / f
    s = p.read_text()
    for a, b in pairs:
        assert s.count(a) == 1, (f, s.count(a), a[:140])
        s = s.replace(a, b)
    p.write_text(s)


rep("plan8.py", [
    ("""RRTWIN = os.environ.get("RRTWIN") == "1"  # round 25 timing twin (outputs wrong): norms skip their residual input (a norm's output)
""",
     """RRTWIN = os.environ.get("RRTWIN") == "1"  # round 25 timing twin (outputs wrong): norms skip their residual input (a norm's output)
RRES = tuple(x for x in os.environ.get("RRES", "").split(",") if x)  # round 25: 3-input norms whose residual rows rb < RRB come on chip
RRB = int(os.environ.get("RRB", "10"))  # (120 units of 1 KB fit memtile half 1: 124,416 B)
"""),
    ("""        self.ops.append(dict(kind="ew", fn="norm", name=name, mode=mode, n_in=n_in, params_w=params_w, params_bytes=params_bytes,
                             fills=fills, c=drains, dst=dst, srcs=srcs))
""",
     """        self.ops.append(dict(kind="ew", fn="norm", name=name, mode=mode, n_in=n_in, params_w=params_w, params_bytes=params_bytes,
                             fills=fills, c=drains, dst=dst, srcs=srcs))
        if name.split("_", 1)[-1] in RRES and n_in == 3:  # round 25 RRES: stream 2 (the residual, an earlier norm's X') of rb < RRB
            assert srcs[2]["base"] in self.norm_dsts and nres, name  # from memtile half 1 (parked during the producer GEMM)
            base0 = lambda c: srcs[0]["base"] + c * srcs[0]["col"]
            self.ops[-1]["rres"] = RRB
            self.ops[-1]["rfills"] = [[self.tap("act", base0(c), [RRB, 12, 2, XB], [UNIT, KSTEP16, d, 1]),
                                       self.tap("act", base0(c) + RRB * UNIT, [RB - RRB, 12, 3, XB], [UNIT, KSTEP16, d, 1])] for c in range(COLS)]
            self.ops[-1]["rpre"] = [self.tap("act", srcs[2]["base"] + c * srcs[2]["col"], [RRB, 12, 1, XB], [UNIT, KSTEP16, 0, 1])
                                    for c in range(COLS)]
"""),
    ("""\"r3", "msplit", "perm", "tmap", "yxh")""",
     """\"r3", "msplit", "perm", "tmap", "yxh", "rres", "rfills", "rpre")"""),
])
rep("pack8.py", [
    ("""def norm_params(mode, scale, beta=None, n_in=1, eps=1e-6, nres=False):""",
     """def norm_params(mode, scale, beta=None, n_in=1, eps=1e-6, nres=False, rres=0):"""),
    ("""    hdr[0] = n_in
    out[2] = bf(eps)
""",
     """    hdr[0] = n_in
    out[2] = bf(eps)
    hdr[2] = rres  # round 25 RRES: row blocks whose stream 2 comes from the EW S2MM 1 ring
"""),
])
rep("step8.py", [
    ("""        plan8.RRTWIN = bool(bj.get("rrtwin", False))  # round 25 timing twin: norms without their residual input
""",
     """        plan8.RRTWIN = bool(bj.get("rrtwin", False))  # round 25 timing twin: norms without their residual input
        plan8.RRES, plan8.RRB = tuple(bj.get("rres", [])), int(bj.get("rrb", 10))  # round 25 RRES: residual rows on chip
"""),
    ("""P.norm_params("rms", sd[key], n_in=op["n_in"], nres=op.get("nres", False)))""",
     """P.norm_params("rms", sd[key], n_in=op["n_in"], nres=op.get("nres", False), rres=op.get("rres", 0)))"""),
    ("""P.norm_params("ln", sd[ln + ".weight"], sd[ln + ".bias"], n_in=op["n_in"], nres=op.get("nres", False)))""",
     """P.norm_params("ln", sd[ln + ".weight"], sd[ln + ".bias"], n_in=op["n_in"], nres=op.get("nres", False), rres=op.get("rres", 0)))"""),
])
rep("design8.py", [
    # EW flags + the R ring type
    ("""    RingP = np.ndarray[(2 * 576 // 4,), np.dtype[np.uint32]]  # 2 slots (L1: the EW stack gives up 1 KB for it, bank 0)
""",
     """    RingP = np.ndarray[(2 * 576 // 4,), np.dtype[np.uint32]]  # 2 slots (L1: the EW stack gives up 1 KB for it, bank 0)
    rres_on = bool(os.environ.get("RRES"))  # round 25 RRES: a 2 x 1 KB ring on the EW core's S2MM 1 (the residual from memtile half 1)
    assert rres_on == ("-DRRES" in ew_flags), "RRES (plan) and -DRRES (EW_DEFS) go together"
    assert not rres_on or nres
    ew_flags += ["-DL_R_FREE=6", "-DL_R_FILL=7"] if rres_on else []
    RingR = np.ndarray[(2 * 1024 // 4,), np.dtype[np.uint32]]
"""),
    ("""        ew_op = Kernel("ew_op", ew_object(ew_flags), [RingIn, RingOut] + ([RingP] if nres else []))""",
     """        ew_op = Kernel("ew_op", ew_object(ew_flags), [RingIn, RingOut] + ([RingP] if nres else []) + ([RingR] if rres_on else []))"""),
    ("""    ew_workers = []
    any_ew = bool(ew_pre or ew_body or ew_post)
""",
     """    r_flow = [Flow(mems[c], ew_tiles[c], dst_channel=1, name=f"Rr{c}") for c in range(COLS)] if rres_on else []
    ew_workers = []
    any_ew = bool(ew_pre or ew_body or ew_post)
"""),
    ("""            core_dmas.append(TileDma(t, p_chan + [
                DmaChannel(DMAChannelDir.S2MM, e_in[c].endpoint(t), [Bd(""",
     """            r_chan, rr = [], None
            if rres_on:  # round 25 RRES: the residual units (1 KB) from the memtile, 2 slots, locks 6 / 7
                rr = Buffer(tile=t, type=RingR, name=f"ewr{c}")
                r_free = Lock(tile=t, init=2, name=f"erf{c}", lock_id=6)
                r_fill = Lock(tile=t, init=0, name=f"eru{c}", lock_id=7)
                r_chan = [DmaChannel(DMAChannelDir.S2MM, r_flow[c].endpoint(t), [Bd(
                    rr, tap=TensorAccessPattern((512,), 0, [1, 1, 1, 256], [0, 0, 0, 1]),
                    acquires=[Acquire(r_free)], releases=[Release(r_fill)], iteration=BdIteration(size=2, stride=256))])]
            core_dmas.append(TileDma(t, p_chan + r_chan + [
                DmaChannel(DMAChannelDir.S2MM, e_in[c].endpoint(t), [Bd("""),
    ("""            def ew_fn_p(rin, rout, rp, ew_op):
                ew_op(rin, rout, rp)

            if nres:""",
     """            def ew_fn_p(rin, rout, rp, ew_op):
                ew_op(rin, rout, rp)

            def ew_fn_pr(rin, rout, rp, rr, ew_op):
                ew_op(rin, rout, rp, rr)

            if rres_on:  # (data_size: .data + .bss = 26,176 B; the 2 KB R ring takes the freed tail, the bank-2 region keeps 4,480 >= 4,160)
                ew_workers.append(Worker(ew_fn_pr, [rin, rout, rp, rr, ew_op], tile=t, stack_size=0x1300,
                                         data_size=int(os.environ.get("RRES_DATA", "26240"))))
            elif nres:"""),
    # memtile: the RF lock and the park / feed tasks
    ("""            if a3:  # round 22 3A: region 3's free / loaded chunks
                lk["E3"] = Lock(tile=mems[c], init=12, name=f"AE3_{c}")
                lk["F3"] = Lock(tile=mems[c], init=0, name=f"AF3_{c}")
            rt_locks.append(lk)""",
     """            if a3:  # round 22 3A: region 3's free / loaded chunks
                lk["E3"] = Lock(tile=mems[c], init=12, name=f"AE3_{c}")
                lk["F3"] = Lock(tile=mems[c], init=0, name=f"AF3_{c}")
            if os.environ.get("RRES"):  # round 25 RRES: half 1 holds a norm's parked residual rows
                lk["RF"] = Lock(tile=mems[c], init=0, name=f"ARF{c}")
            rt_locks.append(lk)"""),
    ("""                rt_tasks.append(t)
        slot_ops = [o for o in ops if o["kind"] in ("gemm", "attn")]""",
     """                if r_flow:  # round 25 RRES: park (DDR -> half 1, all 12 chunks free) and feed (half 1 -> EW S2MM 1, then free)
                    rw = 12 * 256 * RRB_
                    assert rw <= 12 * CH
                    t["rpre"] = me_in.task(Bd(slot, tap=TensorAccessPattern((ra_buf,), 12 * CH, [1, 1, 1, rw], [0, 0, 0, 1]),
                                              acquires=[Acquire(E[1], value=12)], releases=[Release(lk["RF"])]))
                    t["rfeed"] = r_flow[c].endpoint(mems[c]).task(Bd(slot, tap=TensorAccessPattern((ra_buf,), 12 * CH, [1, 1, 1, rw], [0, 0, 0, 1]),
                                                                     acquires=[Acquire(lk["RF"])], releases=[Release(E[1], value=12)]))
                rt_tasks.append(t)
        slot_ops = [o for o in ops if o["kind"] in ("gemm", "attn")]"""),
    ("""        CH = words_a(m, k)  # words per k-step chunk of the slot (144 rows x 64 bfp16 = 2592)
""",
     """        CH = words_a(m, k)  # words per k-step chunk of the slot (144 rows x 64 bfp16 = 2592)
        rres_ops = [o for o in ops if o.get("rres")]  # round 25 RRES
        RRB_ = rres_ops[0]["rres"] if rres_ops else 0
        assert all(o["rres"] == RRB_ for o in rres_ops) and bool(rres_ops) == bool(r_flow), "RRES: plan and design disagree"
"""),
    # the park: pushed by the GEMM right before the RRES norm, after its own A loads and sends
    ("""                    if not (op2 and os.environ.get("OPPRE2_SENDS", "1") == "1"):
                        a_sends(op)
""",
     """                    if not (op2 and os.environ.get("OPPRE2_SENDS", "1") == "1"):
                        a_sends(op)
                    nr_ = rres_before.get(op["name"])
                    if nr_ is not None:  # round 25 RRES: park the next norm's residual rows rb < RRB in half 1 (lock-governed: E[1] = 12
                        for c in range(COLS):  # after the K = 1536 op before this one released half 1)
                            rt_tasks[c]["rpre"].start()
                            fill(a_in[c], nr_["rpre"][c], group=tg, wait=False)  # (tg is finished after this op's drains: done by then)
"""),
    ("""        for op in seq_ops:
            cur_op[0] = op["name"]
            if op["kind"] == "gemm":
                qkv = isinstance(op["c"][0], list)""",
     """        rres_before = {}  # round 25 RRES: the GEMM issued right before each RRES norm parks its residual rows
        for i, o in enumerate(seq_ops):
            if o.get("rres"):
                assert i > 0 and seq_ops[i - 1]["kind"] == "gemm" and seq_ops[i - 1].get("msplit") and seq_ops[i - 1].get("r3"), o["name"]
                rres_before[seq_ops[i - 1]["name"]] = o
        for op in seq_ops:
            cur_op[0] = op["name"]
            if op["kind"] == "gemm":
                qkv = isinstance(op["c"][0], list)"""),
    # the norm: feed task + the main ring's fills in three parts
    ("""                    if op.get("nres"):  # round 20 NRES: the memtile takes the panels (S2MM 5) before the norm can push any
                        for c in range(COLS):
                            rt_tasks[c]["ldn"].start()
""",
     """                    if op.get("nres"):  # round 20 NRES: the memtile takes the panels (S2MM 5) before the norm can push any
                        for c in range(COLS):
                            rt_tasks[c]["ldn"].start()
                    if op.get("rres"):  # round 25 RRES: the parked residual rows to the EW S2MM 1 ring (waits for the park: RF)
                        for c in range(COLS):
                            rt_tasks[c]["rfeed"].start()
"""),
    ("""                        if ms:  # round 23 MSPLIT-P: row blocks 0-5 while the producer computes rows 6-17
                            parts = []
                            for c in range(COLS):
                                f_ = op["fills"][c][1]
                                parts.append((dict(f_, sizes=[MS_RB] + f_["sizes"][1:]),
                                              dict(f_, sizes=[f_["sizes"][0] - MS_RB] + f_["sizes"][1:], off=f_["off"] + MS_RB * f_["strides"][0])))
                            if cppre:  # round 24 CPPRE: both parts configured before the waits, pushed after them
                                st1 = [prefill(e_in[c], parts[c][0], tg, False) for c in range(COLS)]
                                st2 = [prefill(e_in[c], parts[c][1], tg, False) for c in range(COLS)]
                            for n_, g_ in list(pending):
                                if (n_, g_) not in ms:
                                    g_.finish()
                                    pending.remove((n_, g_))
                            for c in range(COLS):
                                st1[c]() if cppre else fill(e_in[c], parts[c][0], group=tg, wait=False)
                            finish_all()
                            for c in range(COLS):
                                st2[c]() if cppre else fill(e_in[c], parts[c][1], group=tg, wait=False)""",
     """                        if ms:  # round 23 MSPLIT-P: row blocks 0-5 while the producer computes rows 6-17
                            parts = []
                            for c in range(COLS):
                                f_ = op["fills"][c][1]
                                if op.get("rres"):  # round 25 RRES: rb 0-5 | 6..RRB-1 with 2 streams, RRB..17 with 3
                                    f2, f3 = op["rfills"][c]
                                    assert f2["sizes"][0] > MS_RB and cppre
                                    parts.append((dict(f2, sizes=[MS_RB] + f2["sizes"][1:]),
                                                  dict(f2, sizes=[f2["sizes"][0] - MS_RB] + f2["sizes"][1:], off=f2["off"] + MS_RB * f2["strides"][0]), f3))
                                    continue
                                parts.append((dict(f_, sizes=[MS_RB] + f_["sizes"][1:]),
                                              dict(f_, sizes=[f_["sizes"][0] - MS_RB] + f_["sizes"][1:], off=f_["off"] + MS_RB * f_["strides"][0])))
                            if cppre:  # round 24 CPPRE: both parts configured before the waits, pushed after them
                                st1 = [prefill(e_in[c], parts[c][0], tg, False) for c in range(COLS)]
                                st2 = [[prefill(e_in[c], p_, tg, False) for p_ in parts[c][1:]] for c in range(COLS)]
                            for n_, g_ in list(pending):
                                if (n_, g_) not in ms:
                                    g_.finish()
                                    pending.remove((n_, g_))
                            for c in range(COLS):
                                st1[c]() if cppre else fill(e_in[c], parts[c][0], group=tg, wait=False)
                            finish_all()
                            for c in range(COLS):
                                if cppre:
                                    for st_ in st2[c]:
                                        st_()
                                else:
                                    fill(e_in[c], parts[c][1], group=tg, wait=False)"""),
    ("""        for t in rt_tasks:
            for task in (*t["ld"], *t["t0"], *[x for x in t["last"] if x is not None], *t["mid"].values(), *[t[x] for x in ("ldn", "ldz", "pa", "pa1") if x in t],""",
     """        for t in rt_tasks:
            for task in (*t["ld"], *t["t0"], *[x for x in t["last"] if x is not None], *t["mid"].values(), *[t[x] for x in ("ldn", "ldz", "pa", "pa1", "rpre", "rfeed") if x in t],"""),
    ("""    for f in (*a_in, *a_out, *(e_in + e_out if any_ew else []), *p_flow):""",
     """    for f in (*a_in, *a_out, *(e_in + e_out if any_ew else []), *p_flow, *r_flow):"""),
    ("""                                                     rrtwin=os.environ.get("RRTWIN") == "1")))""",
     """                                                     rrtwin=os.environ.get("RRTWIN") == "1",
                                                     rres=[x for x in os.environ.get("RRES", "").split(",") if x], rrb=int(os.environ.get("RRB", "10")))))"""),
])
print("RRES patched (plan8, pack8, step8, design8)")
