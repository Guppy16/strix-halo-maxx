// round 15: one-core bench of r15/gen_resc.py's resc against rescale_rb's O loop (mm8.cc, verbatim below)
#include <aie_api/aie.hpp>
#include <stdint.h>
using B64 = aie::vector<bfloat16, 64>;
using A64 = aie::accum<accfloat, 64>;
#define BF(e) ((e).template to_vector<bfloat16>())
extern "C" void resc(float *o, float *osum, const bfloat16 *srow);
constexpr int G = 64;  // guard floats on each side
__attribute__((section(".aie.bank2"))) alignas(64) static float g_o[G + 512 + G];
__attribute__((section(".aie.bank3"))) alignas(64) static float g_s[G + 64 + G];
alignas(64) static float g_ro[512], g_rs[64];
__attribute__((section(".aie.bank1"))) alignas(64) static bfloat16 g_srow[64];
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
static __attribute__((noinline)) void ref(float *orow, float *os, B64 srow) {
  for (int j = 0; j <= 8; j++) {
    float *oj = j < 8 ? orow + j * 64 : os;
    A64 a(aie::load_v<64>(oj));
    B64 hi = BF(a);
    B64 lo = BF(aie::mac(a, hi, (bfloat16)-1.0f));
    A64 pr = aie::mul(hi, srow);
    pr = aie::mac(pr, lo, srow);
    aie::store_v(oj, pr.template to_vector<float>());
  }
}
// x: [1] seed, [2] O scale; y: [0] resc cycles, [1] bad, [2] ref cycles, [3] first, [4] guard changed
extern "C" void apb(int32_t *x, int32_t *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  g_seed = 3u + x[1];
  const float sc = (float)x[2];
  for (int i = 0; i < 64; i++) g_srow[i] = (bfloat16)(0.5f + rnd(0.5f));  // exp(m_old - m_new) in (0, 1]
  for (int i = 0; i < 512; i++) g_ro[i] = rnd(sc);
  for (int i = 0; i < 64; i++) g_rs[i] = 1.0f + rnd(sc);
  uint32_t t0 = 0, t1 = 0;
  for (int rep = 0; rep < 2; rep++) {
    for (int i = 0; i < G + 512 + G; i++) g_o[i] = (i >= G && i < G + 512) ? g_ro[i - G] : -7.0f;
    for (int i = 0; i < G + 64 + G; i++) g_s[i] = (i >= G && i < G + 64) ? g_rs[i - G] : -7.0f;
    t0 = gstamp();
    resc(g_o + G, g_s + G, g_srow);
    t1 = gstamp();
  }
  const uint32_t r0 = gstamp();
  ref(g_ro, g_rs, aie::load_v<64>(g_srow));
  const uint32_t r1 = gstamp();
  int bad = 0, first = -1, guard = 0;
  for (int i = 0; i < 512; i++) if (((uint32_t *)g_o)[G + i] != ((uint32_t *)g_ro)[i]) { if (first < 0) first = i; bad++; }
  for (int i = 0; i < 64; i++) if (((uint32_t *)g_s)[G + i] != ((uint32_t *)g_rs)[i]) { if (first < 0) first = 1000 + i; bad++; }
  for (int i = 0; i < G; i++) guard += (g_o[i] != -7.0f) + (g_o[G + 512 + i] != -7.0f) + (g_s[i] != -7.0f) + (g_s[G + 64 + i] != -7.0f);
  y[0] = (int32_t)(t1 - t0); y[1] = bad; y[2] = (int32_t)(r1 - r0); y[3] = first; y[4] = guard;
}
