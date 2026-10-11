// round 15: one-core bench of r15/gen_rec.py's reck against ew8.cc's recurrence loop (verbatim below, h in [half][cb] layout)
#include <aie_api/aie.hpp>
#include <stdint.h>
using B32 = aie::vector<bfloat16, 32>;
using A32 = aie::accum<accfloat, 32>;
#define BF(e) ((e).template to_vector<bfloat16>())
#define FV(e) ((e).template to_vector<float>())
extern "C" void reck(const bfloat16 *e2, const float *h5, const float *h6, bfloat16 *hb, float *hout, int first_bits, int rest_bits);
static inline void split(const A32 &a, B32 &hi, B32 &lo) { hi = BF(a); lo = BF(aie::mac(a, hi, (bfloat16)-1.0f)); }
// arena layout as in ew8.cc (offsets from g_e2): mult 0, xc 1088, gx (= hb) 2112, a 3136; hout separately
__attribute__((section(".aie.bank1"))) alignas(64) static bfloat16 g_ar[(3136 + 1024 + 512) / 2];
__attribute__((section(".aie.bank3"))) alignas(64) static float g_h[256 + 64 + 256 + 64];  // P[5], a gap, P[6], over-read room
__attribute__((section(".aie.bank2"))) alignas(64) static float g_ho[512 + 64];
alignas(64) static bfloat16 g_rxc[512], g_rgx[512], g_ra[512], g_rm[512];
alignas(64) static float g_rh5[256], g_rh6[256], g_rho[512];
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
static __attribute__((noinline)) void ref(const float *P5, const float *P6, int first, bfloat16 *hb, float *hout) {
  const B32 ones = aie::broadcast<bfloat16, 32>(1.0f);
  const B32 mfirst = aie::broadcast<bfloat16, 32>(first ? 1.0f : 0.0f), mrest = aie::broadcast<bfloat16, 32>(first ? 0.0f : 1.0f);
  for (int cb = 0; cb < 8; cb++) {
#pragma unroll
    for (int half = 0; half < 2; half++) {
      const int i = 2 * cb + half;
      B32 xc = aie::load_v<32>(g_rxc + i * 32);
      B32 gx = aie::load_v<32>(g_rgx + i * 32), a = aie::load_v<32>(g_ra + i * 32), mult = aie::load_v<32>(g_rm + i * 32);
      B32 gated = BF(aie::mul(xc, gx));
      B32 nx = BF(aie::mul(gated, mult));
      A32 h0a(aie::load_v<32>((half ? P6 : P5) + cb * 32));
      B32 h0h, h0l;
      split(h0a, h0h, h0l);
      A32 h = aie::mul(a, h0h);
      h = aie::mac(h, a, h0l);
      h = aie::mac(h, nx, ones);
      B32 hh, hl;
      split(h, hh, hl);
      A32 hsel = aie::mul(hh, mrest);
      hsel = aie::mac(hsel, hl, mrest);
      hsel = aie::mac(hsel, gated, mfirst);
      aie::store_v(hout + half * 256 + cb * 32, FV(hsel));
      aie::store_v(hb + i * 32, BF(hsel));
    }
  }
}
// x: [1] seed, [2] first; y: [0] reck cycles, [1] bad hb, [2] bad hout, [3] C++ cycles, [4] first bad, [5] guard changed
extern "C" void apb(int32_t *x, int32_t *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  g_seed = 5u + x[1];
  const int first = x[2];
  for (int i = 0; i < 512; i++) {
    g_rxc[i] = (bfloat16)rnd(4.0f);
    g_rgx[i] = (bfloat16)(0.5f + rnd(0.5f));
    g_ra[i] = (bfloat16)(0.5f + rnd(0.5f));
    g_rm[i] = (bfloat16)(0.5f + rnd(0.5f));
  }
  for (int i = 0; i < 256; i++) { g_rh5[i] = rnd(8.0f); g_rh6[i] = rnd(8.0f); }
  // the asm's state layout: vector order [cb][half][32]: vector i = 2 cb + half at 128 B i (P[5] holds vectors 0-7)
  float *P5 = g_h, *P6 = g_h + 256 + 64;
  for (int cb = 0; cb < 8; cb++)
    for (int half = 0; half < 2; half++) {
      const int i = 2 * cb + half;
      float *dst = (i < 8 ? P5 + 32 * i : P6 + 32 * (i - 8));
      for (int l = 0; l < 32; l++) dst[l] = (half ? g_rh6 : g_rh5)[cb * 32 + l];
    }
  bfloat16 *e2 = g_ar;
  uint32_t t0 = 0, t1 = 0;
  for (int rep = 0; rep < 2; rep++) {
    for (int i = 0; i < 512; i++) {
      e2[i] = g_rm[i]; e2[544 + i] = g_rxc[i]; e2[1056 + i] = g_rgx[i]; e2[1568 + i] = g_ra[i];
    }
    for (int i = 0; i < 576; i++) g_ho[i] = -7.0f;
    t0 = gstamp();
    reck(e2, P5, P6, e2 + 1056, g_ho, first ? 0x3F80 : 0, first ? 0 : 0x3F80);
    t1 = gstamp();
  }
  static_assert(1088 / 2 == 544 && 2112 / 2 == 1056 && 3136 / 2 == 1568);
  alignas(64) static bfloat16 rhb[512];
  const uint32_t r0 = gstamp();
  ref(g_rh5, g_rh6, first, rhb, g_rho);
  const uint32_t r1 = gstamp();
  int bad_b = 0, bad_h = 0, fb = -1, guard = 0;
  for (int i = 0; i < 512; i++) if (((uint16_t *)(e2 + 1056))[i] != ((uint16_t *)rhb)[i]) { if (fb < 0) fb = i; bad_b++; }
  for (int cb = 0; cb < 8; cb++)
    for (int half = 0; half < 2; half++)
      for (int l = 0; l < 32; l++) {
        const int i = 2 * cb + half;
        if (((uint32_t *)g_ho)[32 * i + l] != ((uint32_t *)g_rho)[half * 256 + cb * 32 + l]) { if (fb < 0) fb = 10000 + 32 * i + l; bad_h++; }
      }
  for (int i = 512; i < 576; i++) guard += g_ho[i] != -7.0f;
  y[0] = (int32_t)(t1 - t0); y[1] = bad_b; y[2] = bad_h; y[3] = (int32_t)(r1 - r0); y[4] = fb; y[5] = guard;
}
