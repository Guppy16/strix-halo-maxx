// round 15: one-core bench of r15/gen_conv.py's convk against ew8.cc's C++ conv loop (verbatim below)
#include <aie_api/aie.hpp>
#include <stdint.h>
#include "aie_kernel_utils.h"
#include "build/r15/convk.h"
using B32 = aie::vector<bfloat16, 32>;
using A32 = aie::accum<accfloat, 32>;
#define BF(e) ((e).template to_vector<bfloat16>())
extern "C" void convk(const bfloat16 *x3, const bfloat16 *x2, const bfloat16 *x1, const bfloat16 *x0, const void *tab, bfloat16 *xc);
constexpr int GUARD = 8;
__attribute__((section(".aie.bank3"))) alignas(64) static bfloat16 g_in[5 * 512];  // 4 parts of 16 vectors + over-read room
__attribute__((section(".aie.bank2"))) alignas(64) static B32 g_tct[8 * 5 + 8];
alignas(64) static B32 g_tcw[32], g_tcb[8];
__attribute__((section(".aie.bank1"))) alignas(64) static bfloat16 g_xc[(16 + 2 * GUARD) * 32];
alignas(64) static bfloat16 g_ref[16 * 32];
static __attribute__((noinline)) uint32_t gstamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}
static uint32_t g_seed;
static float rnd(float s) {
  g_seed = g_seed * 1664525u + 1013904223u;
  return ((int)(g_seed >> 9) - (1 << 22)) * (s / (1 << 22));
}
static __attribute__((noinline)) void ref_conv(const bfloat16 **P, bfloat16 *out) {
  const B32 ones = aie::broadcast<bfloat16, 32>(1.0f);
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int cb = 0; cb < 8; cb++) {
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int off = cb * 64 + h * 32;
      A32 s = aie::mul(BF(aie::mul(aie::load_v<32>(P[4] + off), g_tcw[0 * 8 + cb])), ones);
      s = aie::mac(s, BF(aie::mul(aie::load_v<32>(P[3] + off), g_tcw[1 * 8 + cb])), ones);
      s = aie::mac(s, BF(aie::mul(aie::load_v<32>(P[2] + off), g_tcw[2 * 8 + cb])), ones);
      s = aie::mac(s, BF(aie::mul(aie::load_v<32>(P[1] + off), g_tcw[3 * 8 + cb])), ones);
      aie::store_v(out + off, BF(aie::mac(A32(aie::mul(BF(s), ones)), g_tcb[cb], ones)));
    }
  }
}
// x: [1] seed, [2] x scale (quarters), [3] w scale (quarters); y: [0] convk cycles, [1] bad, [2] C++ cycles, [3] first, [4] got,
// [5] want, [6] guard changed
extern "C" void apb(int32_t *x, int32_t *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  g_seed = 11u + x[1];
  const float xs = x[2] * 0.25f, ws = x[3] * 0.25f;
  for (int i = 0; i < 5 * 512; i++) g_in[i] = (bfloat16)rnd(xs);
  alignas(64) bfloat16 v[32];
  for (int k = 0; k < 40; k++) {
    for (int l = 0; l < 32; l++) v[l] = (bfloat16)rnd(ws);
    if (k < 32) g_tcw[k] = aie::load_v<32>(v); else g_tcb[k - 32] = aie::load_v<32>(v);
  }
  for (int cb = 0; cb < 8; cb++) {
    for (int t = 0; t < 4; t++) g_tct[5 * cb + t] = g_tcw[8 * t + cb];
    g_tct[5 * cb + 4] = g_tcb[cb];
  }
  const bfloat16 *P[5] = {nullptr, g_in + 3 * 512, g_in + 2 * 512, g_in + 512, g_in};  // P[4] = x-3 ... P[1] = x
  bfloat16 *xc = g_xc + GUARD * 32;
  uint32_t t0 = 0, t1 = 0;
  for (int rep = 0; rep < 2; rep++) {
    for (int i = 0; i < (16 + 2 * GUARD) * 32; i++) ((uint16_t *)g_xc)[i] = 0x7FC1;
    t0 = gstamp();
    convk(P[4], P[3], P[2], P[1], g_tct, xc);
    t1 = gstamp();
  }
  const uint32_t r0 = gstamp();
  ref_conv(P, g_ref);
  const uint32_t r1 = gstamp();
  int bad = 0, first = -1, guard = 0;
  for (int i = 0; i < 16 * 32; i++) {
    const uint16_t a = ((uint16_t *)xc)[i], b = ((uint16_t *)g_ref)[i];
    if (a != b) { if (first < 0) { first = i; y[4] = a; y[5] = b; } bad++; }
  }
  for (int i = 0; i < GUARD * 32; i++) guard += (((uint16_t *)g_xc)[i] != 0x7FC1) + (((uint16_t *)(xc + 16 * 32))[i] != 0x7FC1);
  y[0] = (int32_t)(t1 - t0); y[1] = bad; y[2] = (int32_t)(r1 - r0); y[3] = first; y[6] = guard;
}
