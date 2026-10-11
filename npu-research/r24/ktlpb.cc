// pass 19 KTLP-NL: one-core, lock-free bench of the KTL kernel (r24/mkvar.py variants) against the deployed E7 R 68 kernel.
// Static A slots (3 x 1,152 B) and B buffers (2 x 4,608 B, X = Y + 4,032 contiguous), C pieces in g_c (rb 0-11 pair 0) / g_hi.
// Lock ops are alu adds in the variants: nothing can block. x[0] = mode (0: E7 per k-step, 1: KTL per k-step (main's KTL1),
// 2: KTL one call per tile), x[1] = k-steps. y[0..35] piece hashes, y[40 + 16 k ..] the KTL variant's logs per call.
#include <aie_api/aie.hpp>
#include <stdint.h>
#include "e7.h"
#include "e7k.h"
extern "C" void mm_e7_nl(const int32_t *rec, const void *py, float *piece0, const void *px, int32_t cfg);
extern "C" void mm_e7k_nl(const int32_t *rec, float *pc, int32_t *log, int32_t cfg, int32_t trips, int32_t y, int32_t x);
alignas(64) static uint8_t g_a[3][1152];
alignas(64) static uint8_t g_b[2][4608];
alignas(64) static float g_c[3072];
alignas(64) static float g_hi[6144];
alignas(64) static int32_t g_rec[MM_E7K_NHEAD + MM_E7K_NPER + MM_E7_NREC];
alignas(64) static int32_t g_log[64];
alignas(64) static uint8_t g_lb[64][72];  // logical B panels (pair4 index)

static int32_t A(const void *p) { return (int32_t)(uintptr_t)p; }
static int32_t piece(int rb, int pair) {
  if (pair) return A(g_hi) + 1024 * (6 + rb);
  return rb < 12 ? A(g_c) + 1024 * rb : A(g_hi) + 1024 * (rb - 12);
}
static int32_t a_ptr(int r) { return r < 18 ? A(g_a[(r / 2) % 3]) + 576 * (r % 2) : A(g_a[1]); }
static int32_t rec_rot(int r, int f) {
  if (f == 0) return a_ptr(r);
  if (f == 1) return r == 0 ? 0 : r < 18 ? piece(r, 0) - piece(r - 1, 1) - 1024 : r == 18 ? piece(0, 0) - piece(17, 1) - 1024 : 0;
  if (f == 2) return r < 18 ? piece(r, 1) - piece(r, 0) - 1024 : 0;
  if (f == 3) return r < 18 && r % 2 == 0;
  return r < 18 && r % 2 == 1;
}
static int32_t rec_k(int r, int f) {
  r %= 18;
  if (f == 0) return a_ptr(r);
  if (f == 1) return piece(r, 0) - piece((r + 17) % 18, 1) - 1024;
  if (f == 2) return piece(r, 1) - piece(r, 0) - 1024;
  if (f == 3) return r % 2 == 0;
  if (f == 4) return r % 2 == 1;
  return r == 17;
}
static void fill_b(const int8_t *border) {  // object position i holds logical panel border[i]
  for (int b = 0; b < 2; b++)
    for (int i = 0; i < 64; i++)
      for (int j = 0; j < 72; j++) g_b[b][72 * i + j] = g_lb[border[i]][j];
}
extern "C" void apb(int32_t *x, int32_t *y) {
  const int mode = x[0], n = x[1];
  uint32_t s = 12345u;
  auto rnd = [&]() { s = s * 1664525u + 1013904223u; return s >> 8; };
  for (int k = 0; k < 3; k++)
    for (int i = 0; i < 1152; i++) g_a[k][i] = (i % 9 == 0) ? (uint8_t)(120 + rnd() % 8) : (uint8_t)rnd();
  for (int p = 0; p < 64; p++)
    for (int j = 0; j < 72; j++) g_lb[p][j] = (j % 9 == 0) ? (uint8_t)(120 + rnd() % 8) : (uint8_t)rnd();
  for (int i = 0; i < 3072; i++) g_c[i] = 0.f;
  for (int i = 0; i < 6144; i++) g_hi[i] = 0.f;
  for (int i = 0; i < 64; i++) g_log[i] = -7;
  int32_t *rk = g_rec, *re = g_rec + MM_E7K_NHEAD + MM_E7K_NPER;
  for (int i = 0; i < MM_E7K_NHEAD + MM_E7K_NPER; i++) rk[i] = rec_k(mm_e7k_order[i][0], mm_e7k_order[i][1]);
  for (int i = 0; i < MM_E7_NREC; i++) re[i] = rec_rot(mm_e7_order[i][0], mm_e7_order[i][1]);
  float *pc = (float *)(uintptr_t)(piece(0, 0) - rec_k(0, 1));
  if (mode == 0) {
    fill_b(mm_e7_border);
    for (int k = 0; k < n; k++) mm_e7_nl(re, g_b[k % 2], g_c, g_b[k % 2] + 4032, k == 0 ? 781 : 780);
  } else {
    fill_b(mm_e7k_border);
    if (mode == 1)
      for (int k = 0; k < n && k < 3; k++)
        mm_e7k_nl(rk, pc, g_log + 16 * k, k == 0 ? 781 : 780, 18, A(g_b[k % 2]), A(g_b[k % 2] + 4032));
    else
      mm_e7k_nl(rk, pc, g_log, 781, 18 * n, A(g_b[0]), A(g_b[0] + 4032));
  }
  for (int rb = 0; rb < 18; rb++)
    for (int pair = 0; pair < 2; pair++) {
      const uint32_t *w = (const uint32_t *)(uintptr_t)piece(rb, pair);
      uint32_t h = 2166136261u;
      for (int i = 0; i < 256; i++) h = (h ^ w[i]) * 16777619u;
      y[2 * rb + pair] = (int32_t)h;
    }
  y[36] = mode;
  y[37] = A(g_b[0]);
  y[38] = A(g_b[1]);
  y[39] = A(g_rec);
  for (int i = 0; i < 64; i++) y[40 + i] = g_log[i];
}
