// One-core bench of the GEMM k-step (round 9): mm8.cc's mm_kstep (MM_KS: the asm mm_ks; otherwise the C++ loop over
// mm_core_p) on deterministic pseudo-random A / B panels, no DMA. The A ring's locks (aF = 4, aU = 5) are IRON locks of
// this tile; the bench re-arms them around every call (release aU by 9, take aF's 9 back), so no acquire can block.
// Output: per accumulator piece (rb, pair) a hash of its 256 fp32 words, the cycles of the timed calls.
#include "../mm8.cc"

#ifndef KSB_SEED
#define KSB_SEED 12345u
#endif
alignas(64) static uint8_t ksb_a[3][1152];
alignas(64) static uint8_t ksb_b[4608];
alignas(64) static bfloat16 ksb_c[DIM_M * DIM_N];

static inline __attribute__((always_inline)) uint32_t ksb_stamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}

extern "C" void ksb(int32_t *x, int32_t *y) {
  // bfp16ebs8 panels: random mantissas, exponents in a narrow range (finite, no overflow over 9 k-steps)
  uint32_t s = KSB_SEED;
  auto rnd = [&]() { s = s * 1664525u + 1013904223u; return s >> 8; };
  for (int i = 0; i < 3 * 1152; i++) ksb_a[i / 1152][i % 1152] = (i % 9 == 0) ? (uint8_t)(120 + rnd() % 8) : (uint8_t)rnd();
  for (int i = 0; i < 4608; i++) ksb_b[i] = (i % 9 == 0) ? (uint8_t)(120 + rnd() % 8) : (uint8_t)rnd();
  const int n = x[0];
  acc_zero(ksb_c);
  uint32_t t0 = 0, t1 = 0;
  for (int c = 0; c < n; c++) {
    release(48 + 5, 9);  // nine full A slots
    if (c == n - 1) t0 = ksb_stamp();
    mm_kstep((bfp16ebs8 *)ksb_a[0], (bfp16ebs8 *)ksb_a[1], (bfp16ebs8 *)ksb_a[2], (bfp16ebs8 *)ksb_b, ksb_c);
    if (c == n - 1) t1 = ksb_stamp();
    acquire_greater_equal(48 + 4, 9);  // the nine releases back
  }
  for (int rb = 0; rb < DIM_M / 8; rb++)
    for (int pair = 0; pair < 2; pair++) {
#ifdef MM_PIECES
      const uint32_t *w = (const uint32_t *)piece(ksb_c, rb, pair);
#else
      const uint32_t *w = (const uint32_t *)(acc_rows(ksb_c, rb * 8) + 256 * pair);
#endif
      uint32_t h = 2166136261u;
      for (int i = 0; i < 256; i++) h = (h ^ w[i]) * 16777619u;
      y[2 + 2 * rb + pair] = (int32_t)h;
    }
  y[0] = (int32_t)(t1 - t0);
  y[1] = n;
}
