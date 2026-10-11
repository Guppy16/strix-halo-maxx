"""Hand-scheduled AIE2P assembly for design8's GEMM call (round 7, npu-research P18 first step).

  extern "C" void mm_core(const bfp16ebs8 *pA /*p0*/, const bfp16ebs8 *pB /*p1*/, float *pC /*p2*/);

Same arithmetic as mm8.cc's C++ mm_acc (DIM_M 144, DIM_N 64, DIM_K 64, RHO 9): one 16-row stripe = 2 row blocks of 8
(A: [rb 2][q 8] panels, popped through the lda FIFO), C rows += A x B over k = 64, the fp32 accumulators of the stripe
at pC as [rb 2][col block 8][64]. Per accumulator the 8 k-step MACs run in order q = 0..7 from the loaded value, as in
the C++ loop: bit-identical results.

Difference: 4 accumulator chains instead of 2. A group = (row block, pair of 16-column blocks) = 4 col blocks x 8 q =
32 MACs on dm0..dm3, issued one per cycle in the order (q, col block): each chain is 4 cycles apart (the stage-3 rule
needs >= 3; round 6's loop alternated 2 chains at distance 3 = 3 bundles per 2 MACs). That needs the B panels of a
pair in pop order [q][4 col blocks]: pack8 B_PAIR4 (a host permutation inside each 4,608 B weight object).
B pops (ldb, one fill per 8 pops) land at +8 and are read by their MAC exactly then (a 4-register conveyor); the row
block's 8 A panels sit in registers; accumulator quarters load on lda (+7, read at the MAC's stage 3) and store on st
(>= 6 after the chain's last MAC). Group starts S_G are searched upward until every lda deadline is met.

usage: python3 r7/gen_mm.py out.s
"""

import sys

AREG = [0, 1, 2, 3, 4, 6, 8, 10]  # A panel registers (as phase 11 / Peano)
BREG = [5, 7, 9, 11]  # B conveyor
Q = ["bmll", "bmlh", "bmhl", "bmhh"]
NG = 4  # groups per call: (rb, pair) = (0,0) (0,1) (1,0) (1,1)
POP_LAT, LOAD_LAT, ACC_READ, STORE_AFTER = 8, 7, 3, 6


class Infeasible(Exception):
    def __init__(self, group):
        self.group = group


def pops_of(S):
    """ldb ops of a group starting at S: a fill before each 8 pops; returns (fill cycles, pop cycles[32])"""
    fills, pops = [], []
    for j in range(32):
        if j % 8 == 0:
            fills.append(S + 9 * (j // 8))
        pops.append(S + 1 + j + j // 8)
    return fills, pops


SPLIT = False  # bench variant (r8/mmb): mm_core_split(pA, pB, pC_load p2, pC_store p3) - loads and stores apart
# round 9 (npu-research P20 stage a): mm_core_p(pA p0, pB p1, piece0 p2, d1 r0, d2 r1, d3 r2): group G's accumulators
# live in their own 1 KB piece, piece G = piece G-1 + 1 KB + dG; the load / store cursors jump by dG between groups
# (padda on the lda slot, inside the group where lda is free). The C++ side places consecutive pieces in different banks.
PIECES = False


def build(S):
    sched = {}

    def put(c, slot, text):
        sched.setdefault(c, [])
        assert all(s != slot for s, _ in sched[c]), f"slot {slot} double-booked at {c}: {text} vs {sched[c]}"
        sched[c].append((slot, text))

    def busy(c, slot):
        return any(s == slot for s, _ in sched.get(c, []))

    put(0, "lda", "mova r3, #780")
    put(0, "st", "movs p4, p3" if SPLIT else "movs p3, p2")
    put(0, "mv", "mov r24, #0")
    put(1, "mv", "mov r25, #0")
    put(1, "st", "movs p3, p2" if SPLIT else "movs p4, p2")
    put(2, "st", "movs p5, p1")
    if PIECES:
        for G in range(1, NG):
            put(2 + G, "st", f"movs m{G}, r{G - 1}")
    mac = {}  # (G, j) -> cycle
    pop = {}
    for G in range(NG):
        fills, pps = pops_of(S[G])
        for f in fills:
            put(f, "ldb", "vldb.fill.512 [p1, lf1, r25]")
        for j, c in enumerate(pps):
            put(c, "ldb", f"vldb.pop.576 ex{BREG[j % 4]}, [p1, lf1, r25]")
            pop[G, j] = c
            mac[G, j] = c + POP_LAT
            put(c + POP_LAT, "vec", f"vmac.f dm{j % 4}, dm{j % 4}, ex{AREG[j // 4]}, ex{BREG[j % 4]}, r3")
        if G == 1:  # the B stream back to pB for row block 1 (r25 written at +2, after the last pop read it)
            last = pps[-1]
            put(last, "mv", "mov r25, #0")
            put(last + 1, "mv", "mov p1, p5")
            if S[2] < last + 3:
                raise Infeasible(2)
    # stores: in memory order (cursor p4), each quarter >= the chain's last MAC + 6
    store = {}
    c = 3
    for G in range(NG):
        for i in range(4):
            for qq in range(4):
                c = max(c, mac[G, 28 + i] + STORE_AFTER)
                while busy(c, "st"):
                    c += 1
                put(c, "st", f"vst {Q[qq]}{i}, [p4], #64")
                store[G, i, qq] = c
                c += 1
    # lda: two ordered streams (accumulator loads in memory order; the A FIFO), earliest-deadline-first
    accs = []
    for G in range(NG):
        for i in range(4):
            for qq in range(4):
                early = 3 if G == 0 else store[G - 1, i, qq] + 2 - LOAD_LAT  # lands after the old value's store read
                dead = mac[G, i] + ACC_READ - LOAD_LAT
                accs.append((early, dead, f"vlda {Q[qq]}{i}, [p3], #64", G))
    astr = [(3, mac[0, 0] - POP_LAT - 1, "vlda.fill.512 [p0, lf0, r24]", 0)]
    for q in range(8):
        astr.append((3, mac[0, 4 * q] - POP_LAT, f"vlda.pop.576 ex{AREG[q]}, [p0, lf0, r24]", 0))
    # row block 1: A_q after row block 0's last read of ex_q (group 1, MAC 4q+3), before group 2's first (MAC 4q)
    last_read0 = mac[1, 3]  # the fill may only start once A_0's last reader is near (all pops follow it)
    astr.append((last_read0 - LOAD_LAT - 1, mac[2, 0] - POP_LAT - 1, "vlda.fill.512 [p0, lf0, r24]", 2))
    for q in range(8):
        astr.append((mac[1, 4 * q + 3] - POP_LAT + 1, mac[2, 4 * q] - POP_LAT, f"vlda.pop.576 ex{AREG[q]}, [p0, lf0, r24]", 2))
    streams = [accs, astr]
    heads = [0, 0]
    c = 3
    r24_reset = None
    while any(h < len(s) for h, s in zip(heads, streams)):
        cands = []
        for k, (h, s) in enumerate(zip(heads, streams)):
            if h < len(s):
                e, d, t, g = s[h]
                if d < c:
                    raise Infeasible(g)
                if e <= c:
                    cands.append((d, k))
        if cands and not busy(c, "lda"):
            _, k = min(cands)
            e, d, t, g = streams[k][heads[k]]
            if k == 1 and g == 2 and "fill" in t:
                r24_reset = c - 1
                if busy(r24_reset, "mv"):
                    c += 1
                    continue
                put(r24_reset, "mv", "mov r24, #0")
            put(c, "lda", t)
            heads[k] += 1
        c += 1
    if PIECES:  # cursor jumps: p3 after group G's last load, p4 after its last store, each >= 3 cycles before the next use
        loads = {}
        for c, ops in sched.items():
            for sl, t in ops:
                if sl == "lda" and t.startswith("vlda bm"):
                    loads.setdefault("all", []).append(c)
        lc = sorted(loads["all"])
        for G in range(NG - 1):
            last_l, first_l = lc[16 * G + 15], lc[16 * (G + 1)]
            last_s, first_s = store[G, 3, 3], store[G + 1, 0, 0]
            for cur, lo, hi in (("p3", last_l + 1, first_l - 3), ("p4", last_s + 1, first_s - 3)):
                c = lo
                while c <= hi and busy(c, "lda"):
                    c += 1
                if c > hi:
                    raise SystemExit(f"no lda slot for the {cur} jump after group {G} in [{lo}, {hi}]")
                put(c, "lda", f"padda [{cur}], m{G + 1}")
    return sched, mac, pop, store


def check(sched, mac, pop):
    """conveyor and A-register reads: each MAC sees exactly the intended pop"""
    pops = {}
    for c, ops in sched.items():
        for sl, t in ops:
            if ".pop.576" in t:
                pops.setdefault(t.split()[1].rstrip(","), []).append(c)
    for (G, j), c in mac.items():
        b = f"ex{BREG[j % 4]}"
        landed = [p for p in pops[b] if p + POP_LAT <= c]
        assert landed and max(landed) == pop[G, j], f"MAC {G},{j} at {c}: {b} last landed pop {landed[-1:]} != {pop[G, j]}"
        a = f"ex{AREG[j // 4]}"
        landed = sorted(p for p in pops[a] if p + POP_LAT <= c)
        want = 0 if G < 2 else 1  # the k-th pop of ex_q is row block k's A_q
        assert len(landed) == want + 1, f"MAC {G},{j} at {c}: {a} has {len(landed)} landed pops, wants row block {want}"


def main(out, name="mm_core"):
    S = [10, 0, 0, 0]
    for G in range(1, NG):
        S[G] = S[G - 1] + 36
    for _ in range(400):
        try:
            sched, mac, pop, store = build(S)
            break
        except Infeasible as e:
            for G in range(e.group, NG):
                S[G] += 1
    else:
        raise SystemExit("no schedule")
    check(sched, mac, pop)
    last = max(sched)
    sched.setdefault(last + 1, []).append(("alu", "ret lr"))
    lines = ["\t.text", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    for c in range(0, last + 2):
        ops = sched.get(c, [])
        lines.append("\tnop" if not ops else "\t" + ";\t\t".join(t for _, t in ops))
    lines += ["\tnop"] * 5 + [f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"{out}: group starts {S}, {last + 7} bundles per call for 128 MACs ({(last + 7) / 128:.2f} per MAC; "
          f"the C++ loop: 321)")


if __name__ == "__main__":
    if "--pieces" in sys.argv:
        sys.argv.remove("--pieces")
        PIECES = True
        main(sys.argv[1], "mm_core_p")
    elif "--split" in sys.argv:
        sys.argv.remove("--split")
        SPLIT = True
        main(sys.argv[1], "mm_core_split")
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "mm_core.s")
