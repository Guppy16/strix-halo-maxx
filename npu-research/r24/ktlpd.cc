// pass 19 KTLP-D: the KTL kernel with REAL locks on one core, A and B rings fed by shim DMAs (deployed lock ids: B prod 0 / cons 1,
// A free 4 / full 5), one call per tile of KTLPD_N k-steps. The caller acquires B(0) only; the kernel takes the rest and releases
// every B and A object. y[0..35] piece hashes (must equal KTLP-NL's lock-free E7 reference), y[36] cycles of the call.
#include <aie_api/aie.hpp>
#include <stdint.h>
#include "e7k.h"
#ifndef KTLPD_N
#define KTLPD_N 3
#endif
extern "C" void mm_e7k(const int32_t *rec, float *pc, int32_t cfg, int32_t trips, int32_t y, int32_t x);
alignas(64) static float g_c[3072];
alignas(64) static float g_hi[6144];
alignas(64) static int32_t g_rec[MM_E7K_NHEAD + MM_E7K_NPER];
static int32_t g_slot[3];
static int32_t A(const void *p) { return (int32_t)(uintptr_t)p; }
static int32_t piece(int rb, int pair) {
  if (pair) return A(g_hi) + 1024 * (6 + rb);
  return rb < 12 ? A(g_c) + 1024 * rb : A(g_hi) + 1024 * (rb - 12);
}
static int32_t rec_k(int r, int f) {
  r %= 18;
  if (f == 0) return g_slot[(r / 2) % 3] + 576 * (r % 2);
  if (f == 1) return piece(r, 0) - piece((r + 17) % 18, 1) - 1024;
  if (f == 2) return piece(r, 1) - piece(r, 0) - 1024;
  if (f == 3) return r % 2 == 0;
  if (f == 4) return r % 2 == 1;
  return r == 17;
}
static inline uint32_t tm() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}
extern "C" void ktlpd(int32_t *y, uint32_t *a0, uint32_t *a1, uint32_t *a2, uint8_t *b0, uint8_t *b1) {
  g_slot[0] = A(a0), g_slot[1] = A(a1), g_slot[2] = A(a2);
  for (int i = 0; i < 3072; i++) g_c[i] = 0.f;
  for (int i = 0; i < 6144; i++) g_hi[i] = 0.f;
  for (int i = 0; i < MM_E7K_NHEAD + MM_E7K_NPER; i++) g_rec[i] = rec_k(mm_e7k_order[i][0], mm_e7k_order[i][1]);
  float *pc = (float *)(uintptr_t)(piece(0, 0) - rec_k(0, 1));
  acquire_greater_equal(b0, 48 + 1, 1);  // B(0): the only B lock the caller takes
  const uint32_t t0 = tm();
  mm_e7k(g_rec, pc, 781, 18 * KTLPD_N, A(b0), A(b0) + 4032);
  const uint32_t t1 = tm();
  for (int rb = 0; rb < 18; rb++)
    for (int pair = 0; pair < 2; pair++) {
      const uint32_t *w = (const uint32_t *)(uintptr_t)piece(rb, pair);
      uint32_t h = 2166136261u;
      for (int i = 0; i < 256; i++) h = (h ^ w[i]) * 16777619u;
      y[2 * rb + pair] = (int32_t)h;
    }
  y[36] = (int32_t)(t1 - t0);
  y[37] = A(b0);
  y[38] = A(b1);
  y[39] = KTLPD_N;
}
