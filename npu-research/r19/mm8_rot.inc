// ROT (npu-research pass 14, E1): the GEMM k-step with a rotating accumulator, r19/gen_rot.py -> build/mm_rot.s / .h.
// Drop-in for mm_kstep's MM_KS path: same arguments, locks (acq.cond #53 / rel.cond #52), pieces (MM_PIECES) and C layout;
// B must be packed in mm_rot_border's order (pack8: B_ROT). Bit-identical to mm_ks: every accumulator gets the same 8 vmacs
// in k-panel order 0..7 on the same panels.
// Include after mm8.cc's MM_PIECES helpers (piece(), rho) with -DMM_ROT; with -DKSC the records are build-time data.
#include "build/mm_rot.h"
extern "C" void mm_rot(const int32_t *rec, const bfp16ebs8 *pB, float *piece0);

// piece(rb, pair) on integer addresses (mm8.cc piece())
constexpr int32_t rot_piece(int32_t c, int32_t hi, int rb, int pair) {
  return pair ? hi + 1024 * (6 + rb) : rb < 12 ? c + 1024 * rb : hi + 1024 * (rb - 12);
}
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
#ifdef KSC
struct RotRec {
  int32_t r[MM_ROT_NREC];
};
static constexpr RotRec rot_rec = [] {
  RotRec t{};
  for (int i = 0; i < MM_ROT_NREC; i++)
    t.r[i] = rot_record(mm_rot_order[i][0], mm_rot_order[i][1], (int32_t)KS_C, (int32_t)KS_ACC_HI, (int32_t)KS_A0,
                        (int32_t)KS_A1, (int32_t)KS_A2);
  return t;
}();
static inline void rot_kstep(bfp16ebs8 *, bfp16ebs8 *, bfp16ebs8 *, bfp16ebs8 *pB, bfloat16 *c_out) {
  mm_rot(rot_rec.r, pB, piece(c_out, 0, 0));
}
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
  mm_rot(g_rot_tab, pB, piece(c_out, 0, 0));
}
#endif
