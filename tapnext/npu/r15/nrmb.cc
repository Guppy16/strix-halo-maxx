// round 15: one-core bench of r15/gen_nrm.py's nrm_rms / nrm_ln against norm_rows' pass-2 math (ew8.cc, verbatim below)
#include <aie_api/aie.hpp>
#include <stdint.h>
using B8 = aie::vector<bfloat16, 8>;
using B32 = aie::vector<bfloat16, 32>;
using A32 = aie::accum<accfloat, 32>;
using A64 = aie::accum<accfloat, 64>;
using OutS = aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::none>;
#define BF(e) ((e).template to_vector<bfloat16>())
extern "C" void nrm_rms(const bfloat16 *x, const bfloat16 *p0, bfp16ebs8 *panels, const bfloat16 *rr, bfloat16 *xdst);
extern "C" void nrm_ln(bfloat16 *x, const bfloat16 *p0, bfp16ebs8 *panels, const void *rr, const bfloat16 *p1, bfloat16 *xdst);
__attribute__((section(".aie.bank0"))) alignas(64) static bfloat16 g_xo[512 + 64];
static inline B32 rep4(const bfloat16 *v8) { B8 s = aie::load_v<8>(v8); return aie::concat(s, s, s, s); }
static inline void split32(const A32 &a, B32 &hi, B32 &lo) { hi = BF(a); lo = BF(aie::mac(a, hi, (bfloat16)-1.0f)); }
__attribute__((section(".aie.bank1"))) alignas(64) static bfloat16 g_x[512], g_rx[512];
__attribute__((section(".aie.bank2"))) alignas(64) static bfloat16 g_p0[768], g_p1[768];
__attribute__((section(".aie.bank3"))) alignas(64) static uint8_t g_pan[1024], g_rpan[1024];
alignas(64) static uint8_t g_rr[512];
alignas(64) static bfloat16 g_xin[512];
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
template <bool LN>
static __attribute__((noinline)) void ref(bfloat16 *st, const bfloat16 *p0, const bfloat16 *p1, B32 r0, B32 r1, B32 rl0, B32 rl1, A32 nb0, A32 nb1, uint8_t *pan) {
  const B32 ones = aie::broadcast<bfloat16, 32>(1.0f);
  OutS ps((bfp16ebs8 *)pan);
  for (int cb = 0; cb < 8; cb++) {
    B32 x0 = aie::load_v<32>(st + cb * 64), x1 = aie::load_v<32>(st + cb * 64 + 32);
    B32 z0, z1;
    if constexpr (LN) {
      B32 ga = rep4(p0 + cb * 8), be = rep4(p1 + cb * 8);
      A32 t0 = nb0, t1 = nb1;
      t0 = aie::mac(t0, x0, r0);
      t0 = aie::mac(t0, x0, rl0);
      t1 = aie::mac(t1, x1, r1);
      t1 = aie::mac(t1, x1, rl1);
      B32 th, tl;
      split32(t0, th, tl);
      A32 y0 = aie::mul(be, ones);
      y0 = aie::mac(y0, th, ga);
      y0 = aie::mac(y0, tl, ga);
      split32(t1, th, tl);
      A32 y1 = aie::mul(be, ones);
      y1 = aie::mac(y1, th, ga);
      y1 = aie::mac(y1, tl, ga);
      z0 = BF(y0);
      z1 = BF(y1);
      aie::store_v(st + cb * 64, z0);
      aie::store_v(st + cb * 64 + 32, z1);
    } else {
      B32 sc = rep4(p0 + cb * 8);
      z0 = BF(aie::mul(BF(aie::mul(x0, r0)), sc));
      z1 = BF(aie::mul(BF(aie::mul(x1, r1)), sc));
    }
    ps.push(A64(aie::concat(z0, z1)).template to_vector<bfp16ebs8>());
  }
}
// x: [1] seed, [2] mode (0 rms, 1 ln), [3] q (param offset); y: [0] asm cycles, [1] bad panel bytes, [2] bad z, [3] C++ cycles, [4] first
extern "C" void apb(int32_t *x, int32_t *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  g_seed = 9u + x[1];
  const int ln = x[2], q = x[3];
  for (int i = 0; i < 512; i++) g_rx[i] = (bfloat16)rnd(6.0f);
  for (int i = 0; i < 512; i++) g_xin[i] = g_rx[i];
  for (int i = 0; i < 768; i++) { g_p0[i] = (bfloat16)(1.0f + rnd(0.5f)); g_p1[i] = (bfloat16)rnd(0.3f); }
  alignas(64) bfloat16 v[32];
  B32 rv[4];
  for (int k = 0; k < 4; k++) {  // r0, r1, rl0, rl1 (per row: 4 rows x 8 lanes)
    for (int l = 0; l < 32; l++) v[l] = (bfloat16)(k < 2 ? 0.5f + rnd(0.4f) : rnd(0.002f));
    rv[k] = aie::load_v<32>(v);
    aie::store_v((bfloat16 *)(g_rr + 64 * k), rv[k]);
  }
  alignas(64) float nbv[64];
  for (int l = 0; l < 64; l++) nbv[l] = rnd(1.5f);
  for (int l = 0; l < 64; l++) ((float *)(g_rr + 256))[l] = nbv[l];
  const A32 nb0(aie::load_v<32>(nbv)), nb1(aie::load_v<32>(nbv + 32));
  uint32_t t0 = 0, t1 = 0;
  for (int rep = 0; rep < 2; rep++) {
    for (int i = 0; i < 512; i++) g_x[i] = g_rx[i];
    for (int i = 0; i < 1024; i++) g_pan[i] = 0xA5;
    for (int i = 0; i < 576; i++) ((uint16_t *)g_xo)[i] = 0x7FC1;
    t0 = gstamp();
    if (ln) nrm_ln(g_x, g_p0 + q * 64, (bfp16ebs8 *)g_pan, g_rr, g_p1 + q * 64, g_xo);
    else nrm_rms(g_x, g_p0 + q * 64, (bfp16ebs8 *)g_pan, (const bfloat16 *)g_rr, g_xo);
    t1 = gstamp();
  }
  for (int i = 0; i < 1024; i++) g_rpan[i] = 0xA5;
  const uint32_t r0 = gstamp();
  if (ln) ref<true>(g_rx, g_p0 + q * 64, g_p1 + q * 64, rv[0], rv[1], rv[2], rv[3], nb0, nb1, g_rpan);
  else ref<false>(g_rx, g_p0 + q * 64, g_p1 + q * 64, rv[0], rv[1], rv[2], rv[3], nb0, nb1, g_rpan);
  const uint32_t r1 = gstamp();
  int bp = 0, bz = 0, first = -1;
  for (int i = 0; i < 1024; i++) if (g_pan[i] != g_rpan[i]) { if (first < 0) first = i; bp++; }
  for (int i = 0; i < 512; i++) if (((uint16_t *)g_x)[i] != ((uint16_t *)g_rx)[i]) { if (first < 0) first = 10000 + i; bz++; }
  int past = 0, badx = 0;
  for (int i = 0; i < 512; i++) badx += ((uint16_t *)g_xo)[i] != ((uint16_t *)g_xin)[i];
  for (int i = 512; i < 576; i++) badx += ((uint16_t *)g_xo)[i] != 0x7FC1;
  for (int i = 576; i < 1024; i++) past += g_pan[i] != 0xA5;
  y[0] = (int32_t)(t1 - t0); y[1] = bp; y[2] = bz; y[3] = (int32_t)(r1 - r0); y[4] = first; y[5] = past; y[6] = badx;
}
