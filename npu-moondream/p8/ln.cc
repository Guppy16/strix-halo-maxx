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

alignas(64) static bfloat16 g_gamma_b[D], g_beta_b[D];
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
  for (int i = 0; i < D / 32; i++) {
    aie::store_v(g_gamma_b + 32 * i, aie::load_v<32>(p + 32 * i));
    aie::store_v(g_beta_b + 32 * i, aie::load_v<32>(p + D + 32 * i));
  }
}

void ln_rows(bfloat16 *__restrict x, bfloat16 *__restrict y) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  using B32 = aie::vector<bfloat16, 32>;
  const B32 ones = aie::broadcast<bfloat16, 32>(1.0f);
#ifdef LN_RES
  // residual add in bf16: x' = x + delta, written to the second half of the output object and used as the LN input
  for (int r = 0; r < ROWS_PER; r++)
    for (int i = 0; i < D / 32; i++)
      aie::store_v(y + ROWS_PER * D + r * D + 32 * i,
                   aie::add(aie::load_v<32>(x + r * D + 32 * i), aie::load_v<32>(x + ROWS_PER * D + r * D + 32 * i)));
  const bfloat16 *xin = y + ROWS_PER * D;
#else
  const bfloat16 *xin = x;
#endif
  for (int r = 0; r < ROWS_PER; r++) {
    const bfloat16 *xr = xin + r * D;
    // mean: fp32 sum of the bf16 row through the bf16 MAC unit (x * 1, exact)
    aie::accum<accfloat, 32> s0 = aie::zeros<accfloat, 32>(), s1 = s0;
    for (int i = 0; i < D / 64; i++) {
      s0 = aie::mac(s0, aie::load_v<32>(xr + 64 * i), ones);
      s1 = aie::mac(s1, aie::load_v<32>(xr + 64 * i + 32), ones);
    }
    const float mean = aie::reduce_add(aie::add(s0, s1).template to_vector<float>()) / (float)D;
    const bfloat16 mean_b = (bfloat16)mean;
    // variance of (x - mean) with the difference rounded to bf16 (BF16_ELEMENTWISE numerics), fp32 accumulate
    s0 = aie::zeros<accfloat, 32>(); s1 = s0;
    for (int i = 0; i < D / 64; i++) {
      B32 d0 = aie::sub(aie::load_v<32>(xr + 64 * i), mean_b);
      B32 d1 = aie::sub(aie::load_v<32>(xr + 64 * i + 32), mean_b);
      s0 = aie::mac(s0, d0, d0);
      s1 = aie::mac(s1, d1, d1);
    }
    const float var = aie::reduce_add(aie::add(s0, s1).template to_vector<float>()) / (float)D;
    const bfloat16 rstd = (bfloat16)rsqrt_s(var + EPS);
    bfloat16 *yr = y + r * D;
    for (int i = 0; i < D / 32; i++) {
      B32 d = aie::sub(aie::load_v<32>(xr + 32 * i), mean_b);
      B32 v = aie::mul(d, rstd).template to_vector<bfloat16>();
      v = aie::mul(v, aie::load_v<32>(g_gamma_b + 32 * i)).template to_vector<bfloat16>();
      v = aie::add(v, aie::load_v<32>(g_beta_b + 32 * i));
      aie::store_v(yr + 32 * i, v);
    }
  }
  event1();
}
}
