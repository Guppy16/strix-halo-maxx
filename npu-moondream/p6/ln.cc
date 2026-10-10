// LayerNorm over rows of D bf16 values (moondream vision tower: D = 1152), fp32 math:
//   y = (x - mean) * rsqrt(var + eps) * gamma + beta, biased variance, bf16 out (RNE).
// One call = ROWS_PER rows. gamma/beta arrive once as a bf16 pair block [gamma D][beta D].
#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

constexpr int D = LN_D;
constexpr int ROWS_PER = LN_ROWS;
constexpr float EPS = LN_EPS;
static_assert(D % 64 == 0);

alignas(64) static float g_gamma[D], g_beta[D];
using V16 = aie::vector<float, 16>;
#define FV(e) ((e).template to_vector<float>())
// bf16 -> fp32 (exact): through an fp32 accumulator, as the GEMM epilogue does the reverse
static inline V16 bf2f(const bfloat16 *p) { return aie::accum<accfloat, 16>(aie::load_v<16>(p)).template to_vector<float>(); }

// 1/sqrt(x) without libm: bit-trick seed + three Newton steps (fp32 ulp level for x > 0)
static inline float rsqrt_s(float x) {
  union { float f; uint32_t u; } c;
  c.f = x;
  c.u = 0x5f3759dfu - (c.u >> 1);
  float y = c.f;
  for (int i = 0; i < 3; i++) y = y * (1.5f - 0.5f * x * y * y);
  return y;
}

extern "C" {

void ln_params(bfloat16 *p) {
  for (int i = 0; i < D / 16; i++) {
    aie::store_v(g_gamma + 16 * i, bf2f(p + 16 * i));
    aie::store_v(g_beta + 16 * i, bf2f(p + D + 16 * i));
  }
}

void ln_rows(bfloat16 *__restrict x, bfloat16 *__restrict y) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int r = 0; r < ROWS_PER; r++) {
    const bfloat16 *xr = x + r * D;
    // mean (four partial sums for ILP)
    aie::accum<accfloat, 16> s0, s1, s2, s3;
    s0 = aie::zeros<accfloat, 16>(); s1 = s0; s2 = s0; s3 = s0;
    for (int i = 0; i < D / 64; i++) {
      s0 = aie::add(s0, bf2f(xr + 64 * i));
      s1 = aie::add(s1, bf2f(xr + 64 * i + 16));
      s2 = aie::add(s2, bf2f(xr + 64 * i + 32));
      s3 = aie::add(s3, bf2f(xr + 64 * i + 48));
    }
    V16 sv = aie::add(aie::add(FV(s0), FV(s1)), aie::add(FV(s2), FV(s3)));
    const float mean = aie::reduce_add(sv) / (float)D;
    // variance
    s0 = aie::zeros<accfloat, 16>(); s1 = s0; s2 = s0; s3 = s0;
    for (int i = 0; i < D / 64; i++) {
      V16 d0 = aie::sub(bf2f(xr + 64 * i), mean);
      V16 d1 = aie::sub(bf2f(xr + 64 * i + 16), mean);
      V16 d2 = aie::sub(bf2f(xr + 64 * i + 32), mean);
      V16 d3 = aie::sub(bf2f(xr + 64 * i + 48), mean);
      s0 = aie::mac(s0, d0, d0); s1 = aie::mac(s1, d1, d1); s2 = aie::mac(s2, d2, d2); s3 = aie::mac(s3, d3, d3);
    }
    sv = aie::add(aie::add(FV(s0), FV(s1)), aie::add(FV(s2), FV(s3)));
    const float var = aie::reduce_add(sv) / (float)D;
    const float inv = rsqrt_s(var + EPS);
    bfloat16 *yr = y + r * D;
    for (int i = 0; i < D / 16; i++) {
      V16 d = aie::sub(bf2f(xr + 16 * i), mean);
      V16 v = FV(aie::mul(FV(aie::mul(d, inv)), aie::load_v<16>(g_gamma + 16 * i)));
      v = aie::add(v, aie::load_v<16>(g_beta + 16 * i));
      aie::store_v(yr + 16 * i, aie::accum<accfloat, 16>(v).template to_vector<bfloat16>());
    }
  }
  event1();
}
}
