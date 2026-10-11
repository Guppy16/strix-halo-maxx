"""npu-research rev 9: attention K-phase levers priced on the calibrated attsim (r14/attcal.json), each against the deployed
full12r13f as modelled (deltas, ms per step at depth 12). Bit-identical levers first; numerics-changing ones are marked.
    python3 attlev.py
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common" / "tools"))
import attsim as A  # noqa: E402

HERE = Path(__file__).resolve().parent
pb = A.Prm(**json.load(open(HERE / "attcal.json")), )
pb = replace(pb, D=8)
att = A.default_att()
THIN = "--thin" in sys.argv  # primary calibration of rev 9: the map's heavy cores thinned to the in-situ 10.9 %, R 450
if THIN:                     # (per-block check r14/attblk_thin.out: corr 0.97, slope 0.80)
    att = att.thinned(0.109)
    pb = replace(pb, R=450.0)
R13F = dict(a=1, test="flag", exp="flag")


def ms(a, v=None, prm=pb, cnt=None, tail=None):
    c = cnt if cnt is not None else a.counts(v or R13F)
    return 12 * a.block_cycles(c, prm, v or R13F, tail=tail) / A.F / 1e3


base = ms(att)
print(f"model r13f attention {base:.2f} ms/step ({base / 12 * 1e3:.0f} us/block; timeline 10.13 ms); params {pb}")
nolock = ms(att, prm=replace(pb, LOCK=False))
print(f"  lockstep cost today (columns independent): {base - nolock:.2f} ms")
fast_only = ms(att, dict(a=1, test="none", exp="none"))
print(f"  fast loop only (no fallback at all, the F2 twin): {fast_only - base:+.2f} ms")


def row(name, val, note=""):
    print(f"  {val - base:+6.2f} ms  {name}{('  [' + note + ']') if note else ''}")


# --- layouts (bit-identical): deal row blocks to the columns differently ---------------------------------------
def assign_rb_interleave():
    a = np.zeros((8, 3, 6, 3), int)
    for c in range(8):
        for t in range(3):
            for rb in range(6):
                k = 8 * (6 * t + rb) + c  # global row block (natural order: col k // 18, tile (k % 18) // 6, rb k % 6)
                a[c, t, rb] = (k // 18, (k % 18) // 6, k % 6)
    return a


def assign_tile_interleave():
    a = np.zeros((8, 3, 6, 3), int)
    for c in range(8):
        for t in range(3):
            k = 8 * t + c  # global tile (24 per head)
            for rb in range(6):
                a[c, t, rb] = (k // 3, k % 3, rb)
    return a


def assign_pass_balanced():
    """rb interleave that also spreads within a pass: pass 0 tiles get the even rb sweep, pass 1 the rest"""
    return assign_rb_interleave()


att_rbi = att.remap_rows(assign_rb_interleave())
att_ti = att.remap_rows(assign_tile_interleave())
print("\nBit-identical levers:")
row("rows dealt to columns by 8-row block, round robin (Q fill / O drain TAPs)", ms(att_rbi))
row("rows dealt by 48-row tile, round robin", ms(att_ti))
c22 = dict(R13F, j22=False, chunk_L={22: pb.L * 2 / 6 + 40})
row("chunk 22 fast (loop with nkb trips + zero P panels for the invalid blocks)", ms(att, c22))
c0 = dict(R13F, chunk_L={0: 130.0})
row("chunk 0: QK-only loop (gen_attk --qkonly, II 11) instead of the fused pass", ms(att, c0))
c0r = dict(R13F)
cnt = att.counts(c0r)
for p in cnt:  # chunk 0: no rescale (O and the sums are zero)
    n_all, n_l, n_t, n_r, n_e = cnt[p]
    n_r = n_r.copy()
    n_r[..., 0] = 0
    cnt[p] = (n_all, n_l, n_t, n_r, n_e)
row("chunk 0: skip rescale_rb (O, sums are zero)", ms(att, c0r, cnt=cnt), "R is degenerate with T in the fit: 0-400")
row("cheaper slow exp: asm aexp (E 400 -> 250)", ms(att, dict(R13F, e_scale=250 / 400)))
row("cheaper rescale: -O2 / asm rescale_rb (R / 3)", ms(att, dict(R13F, r_scale=1 / 3)))
row("both", ms(att, dict(R13F, e_scale=250 / 400, r_scale=1 / 3)))
row("36-iteration form (L 400 -> 285 per row block: one prologue / preamble per (tile, chunk))", ms(att, dict(R13F, l_scale=285 / 400)))
row("L1: 3 K / V slots per core (needs ~1.1-4.6 KB of L1)", ms(att, prm=replace(pb, SLOTS=3)))
row("L1: 4 K / V slots", ms(att, prm=replace(pb, SLOTS=4)))
cntd, tails = att.deferred(pb)
row("defer flagged units of chunks 1-21 past the lockstep (needs a V copy per deferred chunk: L1)", ms(att, cnt=cntd, tail=tails))

print("\nCombined, bit-identical:")
combo = dict(R13F, j22=False, chunk_L={22: pb.L * 2 / 6 + 40, 0: 130.0})
row("chunk 22 fast + chunk 0 QK-only", ms(att, combo))
row("  + rb interleave", ms(att_rbi, combo))
row("  + asm aexp + cheaper rescale", ms(att_rbi, dict(combo, e_scale=250 / 400, r_scale=1 / 3)))
row("  + 36-iteration form", ms(att_rbi, dict(combo, e_scale=250 / 400, r_scale=1 / 3, l_scale=285 / 400)))

print("\nNumerics-changing (not bit-identical; DAVIS gate), each alone then with chunk 22 fast:")
for pol, note in (("selfseed", "m seeded with q_i . k_i: chunk 0 no longer forced slow; seed cost NOT included"),
                  ("tau96", "TAU 12 nats: exp table to +13 (96 segments)"), ("tau128", "TAU 16 nats: table to +17 (112 segments)"),
                  ("kbinter", "keys dealt to chunks by 8-key block, stride 23")):
    ap = A.Att(HERE / "lazymap.npz", policy=pol)
    if THIN:
        ap = ap.thinned(0.109)
    row(f"{pol}", ms(ap), note)
    row(f"{pol} + chunk 22 fast", ms(ap, c22))
    row(f"{pol} + chunk 22 fast + rb interleave", ms(ap.remap_rows(assign_rb_interleave()), c22))
ap = A.Att(HERE / "lazymap.npz", policy="selfseed")
ap = ap.thinned(0.109) if THIN else ap
row("selfseed + chunk 22 fast + rb interleave + 36-iteration form", ms(ap.remap_rows(assign_rb_interleave()), dict(c22, l_scale=285 / 400)))
row("floor: no fallback at all + 36-iteration form", ms(att, dict(a=1, test="none", exp="none", l_scale=285 / 400)))

print("\nlockstep cost left after each layout (columns independent minus lockstep):")
for name, a in (("today", att), ("rb interleave", att_rbi)):
    print(f"  {name:14s} {ms(a) - ms(a, prm=replace(pb, LOCK=False)):.2f} ms")
