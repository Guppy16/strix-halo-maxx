"""pass 18: EW text levers in norm_rows_k (ew8.cc) behind -DTXT18 (bit-identical by construction: same calls, same order).
  A: the ln / rms pairs of n1k_begin / step / end and nrm_* called through function pointers chosen once per op (one call site
     each instead of two; nrm_rms ignores the 6th pointer, so both share nrm_ln's argument list).
  B: the per-row-block statistics as a 2-trip loop over the halves (one ln_stats / rms_rstd call site each).
usage: python3 npu-research/r23/mk_txt18.py npu-tapnext/ew8.cc OUT.cc"""
import sys

src, out = sys.argv[1], sys.argv[2]
s = open(src).read()
n0 = len(s)


def rep(a, b, count=None):
    global s
    k = s.count(a)
    assert k >= 1, a[:80]
    if count is not None:
        assert k == count, (k, a[:80])
    s = s.replace(a, b)


# A: function pointers, declared after the NIOC locals at the top of norm_rows_k's NPIPE body
rep("""  int ob = g_out_b, osb = g_out_slot_b, ib = g_in_b, iav = g_in_avail, icn = g_in_consumed;
#endif
  for (int rb = -1; rb < RBN; rb++) {""",
    """  int ob = g_out_b, osb = g_out_slot_b, ib = g_in_b, iav = g_in_avail, icn = g_in_consumed;
#endif
#ifdef TXT18  // research pass 18: one call site per kernel (the ln / rms twins have one argument list)
  typedef void (*N1B)(float *, int, int);
  typedef void (*N1S)(const bfloat16 *, const bfloat16 *, const bfloat16 *, bfloat16 *, int, int, int, int);
  typedef void (*N1E)(float *);
  typedef void (*NRM)(bfloat16 *, const bfloat16 *, bfp16ebs8 *, const void *, const bfloat16 *, bfloat16 *);
  const N1B f_begin = ln ? n1k_ln_begin : n1k_rms_begin;
  const N1S f_step = ln ? n1k_ln_step : n1k_rms_step;
  const N1E f_end = ln ? n1k_ln_end : n1k_rms_end;
  const NRM f_nrm = ln ? (NRM)nrm_ln : (NRM)nrm_rms;
#endif
  for (int rb = -1; rb < RBN; rb++) {""", 1)
rep("""          if (ln)
            nrm_ln(sq, g_p0 + q * 64, pd, rr, g_p1 + q * 64, xd);
          else
            nrm_rms(sq, g_p0 + q * 64, pd, (const bfloat16 *)rr, xd);
#ifdef NIO2""", """#ifdef TXT18
          f_nrm(sq, g_p0 + q * 64, pd, rr, ln ? g_p1 + q * 64 : xd, xd);  // nrm_rms: p4 = xdst, p5 unused
#else
          if (ln)
            nrm_ln(sq, g_p0 + q * 64, pd, rr, g_p1 + q * 64, xd);
          else
            nrm_rms(sq, g_p0 + q * 64, pd, (const bfloat16 *)rr, xd);
#endif
#ifdef NIO2""", 1)
rep("""      if (!more) continue;
      if (ln) n1k_ln_begin(g_nacc, w1b, w2b); else n1k_rms_begin(g_nacc, w1b, w2b);""", """      if (!more) continue;
#ifdef TXT18
      f_begin(g_nacc, w1b, w2b);
#else
      if (ln) n1k_ln_begin(g_nacc, w1b, w2b); else n1k_rms_begin(g_nacc, w1b, w2b);
#endif""", 1)
rep("""        if (ln)
          n1k_ln_step(p[0], p[1], p[2], g_stage + q * 512, dc[0], dc[1], dc[2], 64 - IN_B);
        else
          n1k_rms_step(p[0], p[1], p[2], g_stage + q * 512, dc[0], dc[1], dc[2], 64 - IN_B);
#if defined(PROF) || defined(NPROF)
        g_nst += stamp() - ts0;
        g_ncnt++;
#endif
#ifdef NIOC""", """#ifdef TXT18
        f_step(p[0], p[1], p[2], g_stage + q * 512, dc[0], dc[1], dc[2], 64 - IN_B);
#else
        if (ln)
          n1k_ln_step(p[0], p[1], p[2], g_stage + q * 512, dc[0], dc[1], dc[2], 64 - IN_B);
        else
          n1k_rms_step(p[0], p[1], p[2], g_stage + q * 512, dc[0], dc[1], dc[2], 64 - IN_B);
#endif
#if defined(PROF) || defined(NPROF)
        g_nst += stamp() - ts0;
        g_ncnt++;
#endif
#ifdef NIOC""", 1)
rep("""        }
      if (ln) n1k_ln_end(g_nacc); else n1k_rms_end(g_nacc);
    }
    if (!more) continue;""", """        }
#ifdef TXT18
      f_end(g_nacc);
#else
      if (ln) n1k_ln_end(g_nacc); else n1k_rms_end(g_nacc);
#endif
    }
    if (!more) continue;""", 1)
# B: statistics as a 2-trip loop (the NPIPE copy: the first occurrence after "if (!more) continue;\n    if (ln) {")
a = """    if (!more) continue;
    if (ln) {
      B32 r0, r1, rl0, rl1;
      A32 nb0, nb1;
      ln_stats(rowsum8(A32(aie::load_v<32>(g_nacc))), rowsum8(A32(aie::load_v<32>(g_nacc + 64))), eps, r0, rl0, nb0);
      ln_stats(rowsum8(A32(aie::load_v<32>(g_nacc + 32))), rowsum8(A32(aie::load_v<32>(g_nacc + 96))), eps, r1, rl1, nb1);
      aie::store_v((bfloat16 *)rr, r0);
      aie::store_v((bfloat16 *)(rr + 64), r1);
      aie::store_v((bfloat16 *)(rr + 128), rl0);
      aie::store_v((bfloat16 *)(rr + 192), rl1);
      aie::store_v((float *)(rr + 256), FV(nb0));
      aie::store_v((float *)(rr + 384), FV(nb1));
    } else {
      aie::store_v((bfloat16 *)rr, rms_rstd(rowsum8(A32(aie::load_v<32>(g_nacc + 64))), eps));
      aie::store_v((bfloat16 *)(rr + 64), rms_rstd(rowsum8(A32(aie::load_v<32>(g_nacc + 96))), eps));
    }"""
b = """    if (!more) continue;
#ifdef TXT18  // the two halves in a loop: one ln_stats / rms_rstd call site (same calls, same order)
#pragma clang loop unroll(disable)
    for (int h = 0; h < 2; h++) {
      const A32 q = rowsum8(A32(aie::load_v<32>(g_nacc + 64 + 32 * h)));
      if (ln) {
        B32 r, rl;
        A32 nb;
        ln_stats(rowsum8(A32(aie::load_v<32>(g_nacc + 32 * h))), q, eps, r, rl, nb);
        aie::store_v((bfloat16 *)(rr + 64 * h), r);
        aie::store_v((bfloat16 *)(rr + 128 + 64 * h), rl);
        aie::store_v((float *)(rr + 256 + 128 * h), FV(nb));
      } else {
        aie::store_v((bfloat16 *)(rr + 64 * h), rms_rstd(q, eps));
      }
    }
#else
""" + a[len("    if (!more) continue;\n"):] + """
#endif"""
assert s.count(a) == 1, s.count(a)
s = s.replace(a, b)
open(out, "w").write(s)
print(f"{src} -> {out}: +{len(s) - n0} chars")

# C: in_copy and out_copy share one wrap-aware copy (the ring part and the linear part swap with the direction)
s2 = open(out).read()
a_in = """static __attribute__((noinline, minsize)) void in_copy(const uint32_t *ring, void *dst, int off, int bytes) {  // ring -> L1 copy
  int p = g_in_b + off;
  if (p >= IN_B) p -= IN_B;
  const int n = bytes / 64;
  int n1 = (IN_B - p) / 64;  // at most one wrap
  if (n1 > n) n1 = n;
  copy_run((const bfloat16 *)((const char *)ring + p), (bfloat16 *)dst, n1);
  if (n1 < n) copy_run((const bfloat16 *)ring, (bfloat16 *)dst + 32 * n1, n - n1);
}"""
b_in = """#ifdef TXT18  // research pass 18: one wrap-aware copy for both directions (the same copy_run calls, same order)
static __attribute__((noinline, minsize)) void wrap_copy(const uint32_t *ring, int ring_b, int p, bfloat16 *lin, int n, int to_ring) {
  if (p >= ring_b) p -= ring_b;
  int n1 = (ring_b - p) / 64;  // at most one wrap
  if (n1 > n) n1 = n;
  bfloat16 *r = (bfloat16 *)((char *)ring + p), *r0 = (bfloat16 *)ring;
  if (to_ring) copy_run(lin, r, n1); else copy_run(r, lin, n1);
  if (n1 < n) {
    if (to_ring) copy_run(lin + 32 * n1, r0, n - n1); else copy_run(r0, lin + 32 * n1, n - n1);
  }
}
static inline void in_copy(const uint32_t *ring, void *dst, int off, int bytes) {  // ring -> L1 copy
  wrap_copy(ring, IN_B, g_in_b + off, (bfloat16 *)dst, bytes / 64, 0);
}
#else
""" + a_in + """
#endif"""
assert s2.count(a_in) == 1
s2 = s2.replace(a_in, b_in)
a_out = """static __attribute__((noinline)) void out_copy(uint32_t *ring, int off, const bfloat16 *__restrict src, int nvec) {
  int p = g_out_b + off;"""
b_out = """#if defined(TXT18) && defined(CPY64)
static inline void out_copy(uint32_t *ring, int off, const bfloat16 *__restrict src, int nvec) {
  wrap_copy(ring, OUT_B, g_out_b + off, (bfloat16 *)src, nvec, 1);
}
#else
static __attribute__((noinline)) void out_copy(uint32_t *ring, int off, const bfloat16 *__restrict src, int nvec) {
  int p = g_out_b + off;"""
assert s2.count(a_out) == 1
s2 = s2.replace(a_out, b_out)
a_end = """  if (n1 < nvec) COPY_V((bfloat16 *)ring, src + 32 * n1, nvec - n1);
}"""
assert s2.count(a_end) == 1
s2 = s2.replace(a_end, a_end + "\n#endif")
# out_copy is declared before its definition: keep the declaration consistent
a_decl = "static __attribute__((noinline)) void out_copy(uint32_t *ring, int off, const bfloat16 *__restrict src, int nvec);  // (below)"
assert s2.count(a_decl) == 1
s2 = s2.replace(a_decl, "#if !(defined(TXT18) && defined(CPY64))\n" + a_decl + "\n#else\nstatic inline void out_copy(uint32_t *ring, int off, const bfloat16 *__restrict src, int nvec);\n#endif")
open(out, "w").write(s2)
print("C applied")
