// ROT (npu-research pass 14, E1): the GEMM k-step with a rotating accumulator, r19/gen_rot.py -> build/mm_rot.s / .h.
// Drop-in for mm_kstep's MM_KS path: same arguments, locks (acq.cond #53 / rel.cond #52), pieces (MM_PIECES) and C layout;
// B must be packed in mm_rot_border's order (pack8: B_ROT). Bit-identical to mm_ks: every accumulator gets the same 8 vmacs
// in k-panel order 0..7 on the same panels.
// Include after mm8.cc's MM_PIECES helpers (piece(), rho) with -DMM_ROT; with -DKSC the records are build-time data.
#include "build/mm_rot.h"
#if defined(MM_E7K_NPER)  // round 22 KTL (npu-research r22/gen_e7k.py via r22/mkktl.py): one modulo loop over a tile's k-steps; the
// caller acquires B(0), the kernel takes / releases every other B lock itself (mm_ktile)
#ifdef MSPLIT  // round 23 (r23/mm_rot_k70p): + the record period's lda.2d wrap (dn = 6 n - 1, m = -4 dn for n row blocks per k-step)
extern "C" void mm_e7k(const int32_t *rec, float *pc, int32_t cfg, int32_t trips, int32_t y0, int32_t x0, int32_t dn, int32_t m);
#else
extern "C" void mm_e7k(const int32_t *rec, float *pc, int32_t cfg, int32_t trips, int32_t y0, int32_t x0);
#endif
static_assert(MM_E7K_YB == 4032);
#define MM_ROT_NREC (MM_E7K_NHEAD + MM_E7K_NPER)
#define mm_rot_order mm_e7k_order
#elif defined(E7)  // round 21 E7'' (npu-research r22/gen_e7.py via r21/mke7.py): B in the Y buffer (bank 3) and the X buffer (bank 0)
#ifdef MSPLIT  // round 23 MSPLIT-P (r23/mm_rot_e7r68t): the row-block count is the 6th argument
extern "C" void mm_rot(const int32_t *rec, const bfp16ebs8 *pY, float *piece0, const bfp16ebs8 *pX, int32_t conf0, int32_t trips);
#define MM_ROT_CONF0 , nullptr, 780, 18
#else
extern "C" void mm_rot(const int32_t *rec, const bfp16ebs8 *pY, float *piece0, const bfp16ebs8 *pX, int32_t conf0);
#define MM_ROT_CONF0 , nullptr, 780
#endif
static_assert(MM_E7_YB == 4032);
#elif defined(ZLOAD)  // round 21 (r21/mkzload.py): the kp-0 MACs take their config from the 4th argument (781 = zero_acc: a tile's first k-step)
extern "C" void mm_rot(const int32_t *rec, const bfp16ebs8 *pB, float *piece0, int32_t conf0);
#define MM_ROT_CONF0 , 780
#else
extern "C" void mm_rot(const int32_t *rec, const bfp16ebs8 *pB, float *piece0);
#define MM_ROT_CONF0
#endif

// piece(rb, pair) on integer addresses (mm8.cc piece())
constexpr int32_t rot_piece(int32_t c, int32_t hi, int rb, int pair) {
  return pair ? hi + 1024 * (6 + rb) : rb < 12 ? c + 1024 * rb : hi + 1024 * (rb - 12);
}
#ifdef MM_E7K_NPER
// KTL records (npu-research r22/gen_e7k.py record()): any k-step's row block rb mod 18; rb 0's jump continues from rb 17; field 5 = the
// B flag (rb 17: release B(j), acquire B(j + 1), toggle the buffer bases)
constexpr int32_t rot_record(int rb, int f, int32_t c, int32_t hi, int32_t a0, int32_t a1, int32_t a2) {
  const int nrb = 2 * rho;
  rb %= nrb;
  const int32_t slot[3] = {a0, a1, a2};
  switch (f) {
    case 0:
      return slot[(rb / 2) % 3] + 576 * (rb % 2);
    case 1:
      return rot_piece(c, hi, rb, 0) - rot_piece(c, hi, (rb + nrb - 1) % nrb, 1) - 1024;
    case 2:
      return rot_piece(c, hi, rb, 1) - rot_piece(c, hi, rb, 0) - 1024;
    case 3:
      return rb % 2 == 0;
    case 4:
      return rb % 2 == 1;
    default:
      return rb == nrb - 1;
  }
}
// rb 0's (continue) jump: the kernel takes piece(0, 0) minus it, so the prologue's jump lands on piece(0, 0)
constexpr int32_t KTL_J0C = rot_record(0, 1, (int32_t)KS_C, (int32_t)KS_ACC_HI, 0, 0, 0);
#else
// record field f of row block rb (gen_rot.py record()); rb = 2 rho is the dummy row block the last trip starts
constexpr int32_t rot_record(int rb, int f, int32_t c, int32_t hi, int32_t a0, int32_t a1, int32_t a2) {
  const int nrb = 2 * rho;
  const bool real = rb < nrb;
  const int32_t slot[3] = {a0, a1, a2};
  switch (f) {
    case 0:  // A stream start: the stripe's slot, + 576 for its second row block; the dummy reads slot 1 (bank 2)
      return real ? slot[(rb / 2) % 3] + 576 * (rb % 2) : a1;
    case 1:  // p3 / p4 jump into (rb, pair 0); the dummy's loads go to the C object (bank 2) while rb 17 stores to banks 0 / 1
      return rb == 0 ? 0
             : real  ? rot_piece(c, hi, rb, 0) - rot_piece(c, hi, rb - 1, 1) - 1024
             : rb == nrb ? rot_piece(c, hi, 0, 0) - rot_piece(c, hi, nrb - 1, 1) - 1024
                         : 0;
    case 2:  // jump into (rb, pair 1)
      return real ? rot_piece(c, hi, rb, 1) - rot_piece(c, hi, rb, 0) - 1024 : 0;
    case 3:  // acquire the stripe's A slot (even row blocks)
      return real && rb % 2 == 0;
    default:  // release it (odd row blocks)
      return real && rb % 2 == 1;
  }
}
#endif
#ifdef KSC
struct RotRec {
  int32_t r[MM_ROT_NREC];
};
#if defined(MSPLIT) && defined(MM_E7K_NPER)  // (round 23: patched in place for the row passes, mm_ktile)
static RotRec rot_rec = [] {
#else
static constexpr RotRec rot_rec = [] {
#endif
  RotRec t{};
  for (int i = 0; i < MM_ROT_NREC; i++)
    t.r[i] = rot_record(mm_rot_order[i][0], mm_rot_order[i][1], (int32_t)KS_C, (int32_t)KS_ACC_HI, (int32_t)KS_A0,
                        (int32_t)KS_A1, (int32_t)KS_A2);
  return t;
}();
#if defined(MSPLIT) && defined(MM_E7K_NPER)  // round 23 MSPLIT-P on KTL: a row pass (n row blocks from r0) as its own one-k-step table
// record field f of the pass's local row block l (rot_record's KTL fields with the period n; = rot_record for r0 0, n 18)
constexpr int32_t ktl_rec(int l, int f, int r0, int n, int32_t c, int32_t hi, int32_t a0, int32_t a1, int32_t a2) {
  l %= n;
  const int g = r0 + l;
  const int32_t slot[3] = {a0, a1, a2};
  switch (f) {
    case 0:
      return slot[(g / 2) % 3] + 576 * (g % 2);
    case 1:
      return rot_piece(c, hi, g, 0) - rot_piece(c, hi, r0 + (l + n - 1) % n, 1) - 1024;
    case 2:
      return rot_piece(c, hi, g, 1) - rot_piece(c, hi, g, 0) - 1024;
    case 3:
      return g % 2 == 0;
    case 4:
      return g % 2 == 1;
    default:
      return l == n - 1;
  }
}
// the kernel's record order for n row blocks per k-step: the head, then per row block i (i, 5), (i + 1, 3 / 1 / 0), (i, 4), (i + 1, 2)
constexpr int ktl_ord(int i, int n, int which) {
  constexpr int8_t head[4] = {3, 1, 0, 2}, fld[6] = {5, 3, 1, 0, 4, 2}, nxt[6] = {0, 1, 1, 1, 0, 1};
  if (i < 4) return which ? head[i] : 0;
  const int j = i - 4, rb = j / 6, k = j % 6;
  return which ? fld[k] : (rb + nxt[k]) % n;
}
constexpr bool ktl_ord_ok() {
  for (int i = 0; i < MM_ROT_NREC; i++)
    if (mm_rot_order[i][0] != ktl_ord(i, 18, 0) || mm_rot_order[i][1] != ktl_ord(i, 18, 1)) return false;
  return true;
}
static_assert(MM_ROT_NREC == 4 + 6 * 18 && ktl_ord_ok(), "the KTL kernel's record order changed: re-derive ktl_ord");
constexpr int32_t KTL_J0A = ktl_rec(0, 1, 0, 6, (int32_t)KS_C, (int32_t)KS_ACC_HI, 0, 0, 0);
constexpr int32_t KTL_J0B = ktl_rec(0, 1, 6, 12, (int32_t)KS_C, (int32_t)KS_ACC_HI, 0, 0, 0);
// The passes run on rot_rec itself (no L1 for their own tables): pass 1 reads entries 0-39 (head + row blocks 0-5) with entry 34 (rb 5's
// B flag) = 1 and entry 36 (the wrap jump into rb 0) = KTL_J0A; pass 2 reads entries 36-111 with 36-38 as row block 6's head (acquire,
// jump, A pointer) and entry 108 (the wrap jump) = KTL_J0B. ktl_patched() is the table a pass sees; it must equal ktl_rec's.
constexpr int KTL_P[5] = {34, 36, 37, 38, 108};
constexpr int32_t ktl_full(int i) {
  return rot_record(mm_rot_order[i][0], mm_rot_order[i][1], (int32_t)KS_C, (int32_t)KS_ACC_HI, (int32_t)KS_A0, (int32_t)KS_A1,
                    (int32_t)KS_A2);
}
constexpr int32_t ktl_pval(int pass, int k) {  // patched entry KTL_P[k] for pass 0 (whole tiles) / 1 / 2
  return pass == 1 ? (k == 0 ? 1 : k == 1 ? KTL_J0A : ktl_full(KTL_P[k]))
       : pass == 2 ? (k == 0 ? ktl_full(34) : k == 1 ? 1 : k == 2 ? KTL_J0B : k == 3 ? ktl_full(37) : KTL_J0B)
                   : ktl_full(KTL_P[k]);
}
constexpr int32_t ktl_patched(int pass, int i) {
  for (int k = 0; k < 5; k++)
    if (KTL_P[k] == i) return ktl_pval(pass, k);
  return ktl_full(i);
}
constexpr bool ktl_pass_ok(int pass, int r0, int n, int first) {
  for (int i = 0; i < 4 + 6 * n; i++) {
    const int32_t want = ktl_rec(ktl_ord(i, n, 0), ktl_ord(i, n, 1), r0, n, (int32_t)KS_C, (int32_t)KS_ACC_HI, (int32_t)KS_A0,
                                 (int32_t)KS_A1, (int32_t)KS_A2);
    if (ktl_patched(pass, first + i) != want && i != 1) return false;  // (the head jump is the caller's: pc - j0c + it = piece(r0, 0))
  }
  return true;
}
static_assert(ktl_pass_ok(1, 0, 6, 0) && ktl_pass_ok(2, 6, 12, 36), "MSPLIT on KTL: the patched rot_rec is not the pass table");
#endif
#if defined(MSPLIT) && !defined(MM_E7K_NPER)  // round 23 MSPLIT-P: the first row pass (rb 0-5, then the dummy row block 6) has its own records; the second pass
// (rb 6-17) reads rot_rec from row block 6's first record on (its dummy loads from piece(0, 0): reads only)
constexpr int32_t rot_record_n(int rb, int f, int nrb, int32_t c, int32_t hi, int32_t a0, int32_t a1, int32_t a2) {
  const bool real = rb < nrb;
  const int32_t slot[3] = {a0, a1, a2};
  switch (f) {
    case 0:
      return real ? slot[(rb / 2) % 3] + 576 * (rb % 2) : a1;
    case 1:
      return rb == 0 ? 0
             : real  ? rot_piece(c, hi, rb, 0) - rot_piece(c, hi, rb - 1, 1) - 1024
             : rb == nrb ? rot_piece(c, hi, 0, 0) - rot_piece(c, hi, nrb - 1, 1) - 1024
                         : 0;
    case 2:
      return real ? rot_piece(c, hi, rb, 1) - rot_piece(c, hi, rb, 0) - 1024 : 0;
    case 3:
      return real && rb % 2 == 0;
    default:
      return real && rb % 2 == 1;
  }
}
constexpr int MS_NREC = 5 * 6 + 4;
static_assert(MM_ROT_NREC == 5 * 18 + 4 && mm_rot_order[30][0] == 6 && mm_rot_order[30][1] == 3 && mm_rot_order[33][0] == 6 &&
              mm_rot_order[33][1] == 2 && mm_rot_order[34][1] == 4);
struct RotRecA {
  int32_t r[MS_NREC];
};
static constexpr RotRecA rot_rec_a = [] {
  RotRecA t{};
  for (int i = 0; i < MS_NREC; i++)
    t.r[i] = rot_record_n(mm_rot_order[i][0], mm_rot_order[i][1], 6, (int32_t)KS_C, (int32_t)KS_ACC_HI, (int32_t)KS_A0, (int32_t)KS_A1,
                          (int32_t)KS_A2);
  return t;
}();
#endif
#ifdef MM_E7K_NPER  // (mm_kstep is not bound under P35; KTL runs only from mm_ktile)
static inline void rot_kstep(bfp16ebs8 *, bfp16ebs8 *, bfp16ebs8 *, bfp16ebs8 *, bfloat16 *) {}
#else
static inline void rot_kstep(bfp16ebs8 *, bfp16ebs8 *, bfp16ebs8 *, bfp16ebs8 *pB, bfloat16 *c_out) {
  mm_rot(rot_rec.r, pB, piece(c_out, 0, 0) MM_ROT_CONF0);
}
#endif
// with P35 (round 18 mm_ktile): replace `mm_ks(ks_rec.r, b, pc);` in mm_ktile's loop by `mm_rot(rot_rec.r, b, pc);` (both B
// buffers sit in bank 3 as the schedule assumes; mm_rot takes B by pointer)
#else
static int32_t g_rot_tab[MM_ROT_NREC] __attribute__((section(".data.g_rot_tab")));  // .data: the data region's start, never B's bank
static const bfloat16 *g_rot_for = nullptr;
__attribute__((noinline, minsize)) static void rot_records(bfp16ebs8 *const slot[3], bfloat16 *c_out) {
  const int32_t c = (int32_t)(uintptr_t)c_out, hi = (int32_t)(uintptr_t)acc_hi;
#pragma clang loop unroll(disable)
  for (int i = 0; i < MM_ROT_NREC; i++)
    g_rot_tab[i] = rot_record(mm_rot_order[i][0], mm_rot_order[i][1], c, hi, (int32_t)(uintptr_t)slot[0],
                              (int32_t)(uintptr_t)slot[1], (int32_t)(uintptr_t)slot[2]);
  g_rot_for = c_out;
}
static inline void rot_kstep(bfp16ebs8 *a0, bfp16ebs8 *a1, bfp16ebs8 *a2, bfp16ebs8 *pB, bfloat16 *c_out) {
  bfp16ebs8 *const slot[3] = {a0, a1, a2};
  if (c_out != g_rot_for) rot_records(slot, c_out);
  mm_rot(g_rot_tab, pB, piece(c_out, 0, 0) MM_ROT_CONF0);
}
#endif
