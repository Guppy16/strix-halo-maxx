// int8 x int8 -> int32 GEMM microkernel (AIE2P 8x8x8 mmul) with the W8A8
// epilogue of moondream-opt's HI8Lin (hip/i8gemm.hip):
//   C[i][j] = bf16( ((float)acc * sa[j]) * sw[i] + bias[i] )   (+ tanh-GELU)
// Transposed orientation: the per-core tile is C^T[96 weight rows][96 tokens];
// "A" sub-tiles are 16 weight rows x 64 k with that row group's sw / bias
// appended, "B" tiles are 64 k x 96 tokens with the tokens' sa appended.
//
// Layouts (int8 8x8 panels, row-major inside):
//   A sub-tile: [2 row halves][k/8][8 rows x 8 k]  (1024 B) + sw[16] + bias[16] (128 B) = 1152 B
//   B tile:     [n/16][k/8][2][8 k x 8 tokens]       (6144 B) + sa[96] (384 B)           = 6528 B
//   acc / C:    [m/8][n/8][8][8] int32, then bf16 in place (same split as mm_f32acc.cc)
#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

constexpr int m = DIM_M, n = DIM_N, k = DIM_K, rho = RHO;
constexpr int m_a = m / rho;
constexpr int r = 8, s = 8, t = 8;
static_assert(m_a == 16 && n % 16 == 0 && k % 8 == 0 && (m / 2) % 16 == 0);

constexpr int A_PANELS = 2 * (k / s);              // per sub-tile
constexpr int A_BYTES = A_PANELS * 64 + 2 * 16 * 4;
constexpr int B_BYTES = (n / 8) * (k / s) * 64 + n * 4 + 128;  // + [gelu flag, pad]

alignas(64) static int32_t acc_hi[(m / 2) * n];
alignas(64) static float sw_all[m], bias_all[m], sa_all[n];
static int g_gelu = 0;
static int g_stripe = 0;
static volatile int g_zero = 0;

static inline int32_t *acc_rows(bfloat16 *c_out, int row) {
  return row < m / 2 ? (int32_t *)c_out + row * n : acc_hi + (row - m / 2) * n;
}

extern "C" {

void acc_zero(bfloat16 *c_out) {
  const aie::vector<int32_t, 32> z = aie::zeros<int32_t, 32>();
  int32_t *p0 = (int32_t *)c_out;
  for (int i = 0; i < (m / 2) * n / 32; i++) aie::store_v(p0 + 32 * i, z);
  for (int i = 0; i < (m / 2) * n / 32; i++) aie::store_v(acc_hi + 32 * i, z);
  g_stripe = 0;
}

void mm_acc(int8_t *__restrict pA, int8_t *__restrict pB, bfloat16 *c_out) {
  event0();
  using MMUL = aie::mmul<r, s, t, int8, int8>;
  using CPtr = int32_t __aie_dm_resource_c *__restrict;
  const int z = g_zero;
  const int stripe = g_stripe;
  g_stripe = (g_stripe == rho - 1) ? 0 : g_stripe + 1;
  // per-channel params of this 16-row group, per-token scales of this tile
  {
    const float *pp = (const float *)(pA + A_PANELS * 64);
    aie::store_v(sw_all + stripe * m_a, aie::load_v<16>(pp));
    aie::store_v(bias_all + stripe * m_a, aie::load_v<16>(pp + 16));
    const float *ps = (const float *)(pB + (n / 8) * (k / s) * 64);
    for (int i = 0; i < n / 16; i++) aie::store_v(sa_all + 16 * i, aie::load_v<16>(ps + 16 * i));
    g_gelu = ((const int32_t *)ps)[n];
  }
#ifdef NOCOMPUTE
  event1();
  return;
#endif
  for (int rb = 0; rb < 2; rb++) {
    const int8_t *pb = pB;
    CPtr pC = (CPtr)acc_rows(c_out, stripe * m_a + rb * r);
    aie::vector<int32_t, 64> c0 = aie::load_v<64>(pC);
    aie::vector<int32_t, 64> c1 = aie::load_v<64>(pC + r * t);
    AIE_LOOP_MIN_ITERATION_COUNT(n / (2 * t))
#ifdef BLKU
#pragma unroll BLKU
#endif
    for (int blk = 0; blk < n / (2 * t); blk++) {
      CPtr pCb = pC + blk * (2 * r * t);
      CPtr pNext = (blk < n / (2 * t) - 1) ? pCb + 2 * r * t : pCb;
      MMUL acc0((aie::accum<acc32, r * t>(c0)));
      MMUL acc1((aie::accum<acc32, r * t>(c1)));
      c0 = aie::load_v<64>(pNext);
      c1 = aie::load_v<64>(pNext + r * t);
      const int8_t *pa = pA + rb * (k / s) * 64 + z * blk;
#pragma unroll
      for (int q = 0; q < k / s; q++) {
        aie::vector<int8, 64> A0 = aie::load_v<64>(pa + q * 64);
        aie::vector<int8, 64> B0 = aie::load_v<64>(pb);
        aie::vector<int8, 64> B1 = aie::load_v<64>(pb + 64);
        pb += 128;
        acc0.mac(A0, B0);
        acc1.mac(A0, B1);
      }
      aie::store_v(pCb, acc0.template to_vector<int32_t>());
      aie::store_v(pCb + r * t, acc1.template to_vector<int32_t>());
    }
  }
  event1();
}

// Epilogue on native 16-lane fp32 vectors (one vector = two accumulator rows of 8 tokens):
// no 64-lane temporaries, so the GELU polynomial does not spill to the stack.
using V16 = aie::vector<float, 16>;
#define FV(e) ((e).template to_vector<float>())
static inline V16 vmul(V16 a, V16 b) { return FV(aie::mul(a, b)); }
static inline V16 vmuls(V16 a, float b) { return FV(aie::mul(a, b)); }
static inline V16 vadd(V16 a, V16 b) { return aie::add(a, b); }
static inline V16 vadds(V16 a, float b) { return aie::add(a, b); }
static inline V16 vsub(V16 a, V16 b) { return aie::sub(a, b); }

// exp(x) = 2^(x log2 e): degree-6 polynomial on the fraction, integer part into the exponent bits.
static inline V16 exp_v(V16 x) {
  V16 y = vmuls(x, 1.4426950408889634f);
  y = aie::max(aie::min(y, aie::broadcast<float, 16>(126.0f)), aie::broadcast<float, 16>(-126.0f));
  aie::vector<int32_t, 16> ni = aie::to_fixed<int32_t>(vadds(y, -0.5f), 0);
  V16 f = vsub(y, aie::to_float(ni, 0));
  V16 p = aie::broadcast<float, 16>(1.535336188319500e-4f);
  p = vadds(vmul(p, f), 1.339887440266574e-3f);
  p = vadds(vmul(p, f), 9.618437357674640e-3f);
  p = vadds(vmul(p, f), 5.550332471162809e-2f);
  p = vadds(vmul(p, f), 2.402264791363012e-1f);
  p = vadds(vmul(p, f), 6.931472028550421e-1f);
  p = vadds(vmul(p, f), 1.0f);
  aie::vector<int32_t, 16> e = aie::add(aie::upshift(ni, 23), p.template cast_to<int32_t>());
  return e.template cast_to<float>();
}

static inline V16 recip_v(V16 d) {  // 1/d: magic seed + 3 Newton steps
  aie::vector<int32_t, 16> seed = aie::sub(aie::broadcast<int32_t, 16>(0x7EF311C7), d.template cast_to<int32_t>());
  V16 y = seed.template cast_to<float>();
  for (int i = 0; i < 3; i++) y = vmul(y, vsub(aie::broadcast<float, 16>(2.0f), vmul(d, y)));
  return y;
}

static inline V16 gelu_v(V16 x) {
  V16 x3 = vmul(vmul(x, x), x);
  V16 u = vmuls(vadd(x, vmuls(x3, 0.044715f)), 0.7978845608028654f);
  V16 e = exp_v(vmuls(u, 2.0f));
  V16 inv = recip_v(vadds(e, 1.0f));
  V16 tt = vsub(aie::broadcast<float, 16>(1.0f), vmuls(inv, 2.0f));
  return vmul(vmuls(x, 0.5f), vadds(tt, 1.0f));
}

// int32 accumulator tile -> dequant (+bias, +GELU) -> bf16 (RNE), in place.
void acc_store(bfloat16 *c_out) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr int half = (m / 2) * n;
#pragma clang loop unroll(disable)
  for (int ib = 0; ib < m / 8; ib++) {
#pragma clang loop unroll(disable)
    for (int jb = 0; jb < n / 8; jb++) {
      const int off = ib * 8 * n + jb * 64;
      const int32_t *src = off < half ? (const int32_t *)c_out + off : acc_hi + (off - half);
      aie::vector<float, 8> sa8 = aie::load_v<8>(sa_all + jb * 8);
      V16 sa16 = aie::concat(sa8, sa8);
#pragma clang loop unroll(disable)
      for (int h = 0; h < 4; h++) {  // rows 2h, 2h+1 of the block
        const int r0 = ib * 8 + 2 * h;
        V16 v = aie::to_float(aie::load_v<16>(src + 16 * h), 0);
        v = vmul(v, sa16);
        V16 sw16 = aie::concat(aie::broadcast<float, 8>(sw_all[r0]), aie::broadcast<float, 8>(sw_all[r0 + 1]));
        V16 b16 = aie::concat(aie::broadcast<float, 8>(bias_all[r0]), aie::broadcast<float, 8>(bias_all[r0 + 1]));
        v = vadd(vmul(v, sw16), b16);
#if defined(GELU)
        v = gelu_v(v);
#elif defined(GELU_RT)
        if (g_gelu) v = gelu_v(v);
#endif
        aie::store_v(c_out + off + 16 * h, aie::accum<accfloat, 16>(v).template to_vector<bfloat16>());
      }
    }
  }
}
}
