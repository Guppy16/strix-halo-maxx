"""Round 25 RRES: ew8.cc side (EW -DRRES)."""
from pathlib import Path

p = Path.home() / "git/app/.scratch/npu-tapnext/ew8.cc"
s = p.read_text()
R = [
    ("""#ifndef NRES
#define NRES_ARG
#else  // round 20 (research rev 16 NRES): pring != null: the panels go to the 2-slot ring drained by MM2S 1 (-> the memtile A slot)
#define NRES_ARG , uint8_t *pring
#endif
""",
     """#ifndef NRES
#define NRES_ARG
#else  // round 20 (research rev 16 NRES): pring != null: the panels go to the 2-slot ring drained by MM2S 1 (-> the memtile A slot)
#define NRES_ARG , uint8_t *pring
#endif
#ifdef RRES  // round 25: stream 2 (the residual) of row blocks < g_rrb from the 2 x 1 KB ring on S2MM 1 (params header word 2)
#define RRES_ARG , uint32_t *rring
static uint32_t *g_rring;
static int g_rrb;
#else
#define RRES_ARG
#endif
"""),
    ("""      for (int q = q0; q < q0 + NPIPE_G; q++) {
#ifdef NIO2
        const uint32_t tp0 = stamp();
#endif""",
     """#ifdef RRES
      const int rr_on = rb + 1 < g_rrb;  // (pass 1 here is row block rb + 1's)
      const int unit_q = rr_on ? unit - KB : unit;
#else
      const int unit_q = unit;
#endif
      for (int q = q0; q < q0 + NPIPE_G; q++) {
#ifdef NIO2
        const uint32_t tp0 = stamp();
#endif"""),
    ("""        {
          const int d = unit - iav;
          if (d > 0) {
            const int n = div768(d + SLOT_B - 1);
            iav += n * SLOT_B;
            lacq(in_ring, L_IN_FILL, n);
          }
        }
        const int ib0 = ib;""",
     """        {
          const int d = unit_q - iav;
          if (d > 0) {
            const int n = div768(d + SLOT_B - 1);
            iav += n * SLOT_B;
            lacq(in_ring, L_IN_FILL, n);
          }
        }
        const int ib0 = ib;"""),
    ("""          dc[s] = n1 >= 8 ? 0 : 8 - n1;
        }
""",
     """          dc[s] = n1 >= 8 ? 0 : 8 - n1;
        }
#ifdef RRES
        if (rr_on) {
          lacq(g_rring, L_R_FILL, 1);
          p[2] = (const bfloat16 *)((const char *)g_rring + rslot * KB);
          dc[2] = 0;
        }
#endif
"""),
    ("""#ifdef NIOC
        iav -= unit;
        ib += unit;
        if (ib >= IN_B) ib -= IN_B;
        icn += unit;""",
     """#ifdef RRES
        if (rr_on) {
          lrel(g_rring, L_R_FREE, 1);
          rslot ^= 1;
        }
#endif
#ifdef NIOC
        iav -= unit_q;
        ib += unit_q;
        if (ib >= IN_B) ib -= IN_B;
        icn += unit_q;"""),
    ("""  int pslot = 0;                       // (216 units per op: the 2-slot ring is back at slot 0 after every op)
""",
     """  int pslot = 0;                       // (216 units per op: the 2-slot ring is back at slot 0 after every op)
#endif
#ifdef RRES
  int rslot = 0;  // (12 x RRB units per op, an even count: slot 0 again at the next op)
"""),
    ("""  const int n_in = *(const int32_t *)in_at(in_ring, 0);
  const B32 eps""",
     """  const int n_in = *(const int32_t *)in_at(in_ring, 0);
#ifdef RRES
  g_rrb = *(const int32_t *)in_at(in_ring, 8);
#endif
  const B32 eps"""),
    ("""void ew_op(uint32_t *in_ring, uint32_t *out_ring NRES_ARG) {
  in_need(in_ring, 16);""",
     """void ew_op(uint32_t *in_ring, uint32_t *out_ring NRES_ARG RRES_ARG) {
#ifdef RRES
  g_rring = rring;
#endif
  in_need(in_ring, 16);"""),
]
for a, b in R:
    assert s.count(a) == 1, (s.count(a), a[:120])
    s = s.replace(a, b)
p.write_text(s)
print("ew8 RRES patched")
