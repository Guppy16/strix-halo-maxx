// One-core bench of ROT (npu-research pass 14, E1), after npu-tapnext r9/ksb.cc: the deployed k-step (mm8.cc mm_kstep ->
// mm_ks, B in pack8's pair4 order) or mm_rot (B repacked in mm_rot_border's order), same pseudo-random A / B, no DMA.
// The A ring's locks (aF = 4, aU = 5) are re-armed around every call. Output: y[0] cycles of the last call, y[1] calls,
// y[2 + 2 rb + pair] a hash of each accumulator piece (must match between the two builds), y[38] the record count.
#include "mm8.cc"
#include "mm8_rot.inc"

#ifndef KSB_SEED
#define KSB_SEED 12345u
#endif
alignas(64) static uint8_t ksb_a[3][1152];
alignas(64) static uint8_t ksb_b[4608];
// B for mm_rot alone in bank 3 like the deployed B buffers (the linker script's .aie.bank3 holds 4,992 B here)
alignas(64) static uint8_t ksb_br[4608] __attribute__((section(".aie.bank3")));
alignas(64) static bfloat16 ksb_c[DIM_M * DIM_N];

static inline __attribute__((always_inline)) uint32_t ksb_stamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}

extern "C" void rotb(int32_t *x, int32_t *y) {
  uint32_t s = KSB_SEED;
  auto rnd = [&]() { s = s * 1664525u + 1013904223u; return s >> 8; };
  for (int i = 0; i < 3 * 1152; i++) ksb_a[i / 1152][i % 1152] = (i % 9 == 0) ? (uint8_t)(120 + rnd() % 8) : (uint8_t)rnd();
  for (int i = 0; i < 4608; i++) ksb_b[i] = (i % 9 == 0) ? (uint8_t)(120 + rnd() % 8) : (uint8_t)rnd();
#pragma clang loop unroll(disable)
  for (int i = 0; i < 64; i++)  // stream position i holds pair4 panel mm_rot_border[i]
    for (int j = 0; j < 72; j++) ksb_br[72 * i + j] = ksb_b[72 * mm_rot_border[i] + j];
  const int n = x[0];
  acc_zero(ksb_c);
  uint32_t t0 = 0, t1 = 0;
  for (int c = 0; c < n; c++) {
    release(48 + 5, 9);  // nine full A slots
    if (c == n - 1) t0 = ksb_stamp();
#if ROTB_ROT
    rot_kstep((bfp16ebs8 *)ksb_a[0], (bfp16ebs8 *)ksb_a[1], (bfp16ebs8 *)ksb_a[2], (bfp16ebs8 *)ksb_br, ksb_c);
#else
    mm_kstep((bfp16ebs8 *)ksb_a[0], (bfp16ebs8 *)ksb_a[1], (bfp16ebs8 *)ksb_a[2], (bfp16ebs8 *)ksb_b, ksb_c);
#endif
    if (c == n - 1) t1 = ksb_stamp();
    acquire_greater_equal(48 + 4, 9);  // the nine releases back
  }
  for (int rb = 0; rb < DIM_M / 8; rb++)
    for (int pair = 0; pair < 2; pair++) {
      const uint32_t *w = (const uint32_t *)piece(ksb_c, rb, pair);
      uint32_t h = 2166136261u;
      for (int i = 0; i < 256; i++) h = (h ^ w[i]) * 16777619u;
      y[2 + 2 * rb + pair] = (int32_t)h;
    }
#ifdef ROTB_LOCKS  // opt-in diagnostic (UNVERIFIED register offsets: memory-module Lock_value 0x1F000 + 16 id): aF / aU after the
  // calls, both 0 iff mm_rot's rel.cond released exactly 9 per call (rel.cond is new in deployed code; acq.cond is verified)
  y[39] = (int32_t)read_tm(0x1F000 + 16 * 4);
  y[40] = (int32_t)read_tm(0x1F000 + 16 * 5);
#endif
  y[0] = (int32_t)(t1 - t0);
  y[1] = n;
  y[38] = MM_ROT_NREC;
}
