// Primitive characterisation probe (round 6): one core maps every chunk of CH bf16 inputs through one of several
// candidate primitives and stamps the cycles it took. kprobe.py streams all 65,536 bf16 bit patterns through each
// mode; the host compares against exact references (ulp error per input) and reports cycles per 32 lanes.
#include <stdint.h>

#include <aie_api/aie.hpp>

#include "nk8.h"
// core timer, low word (AIE-ML core module Timer_Low, 0x340F8; aie::tile::cycles() needs an undefined get_cycles())
#ifdef NO_STAMP
static inline uint32_t cyc() { return 0; }
#else
static inline uint32_t cyc() { return read_tm(0x340F8); }
#endif

using B32 = aie::vector<bfloat16, 32>;
using A32 = aie::accum<accfloat, 32>;
#define BF(e) ((e).template to_vector<bfloat16>())
#include "gelu_lut.h"

#ifndef REPS
#define REPS 1  // timing builds repeat the mapping REPS times (host-timed: the core timer read hung the device)
#endif

// gelu(x) = x s(x), s = 0.5 (1 + tanh(u)) piecewise-linear from the LUT (16 lanes per lookup); x outside the table
// range: s(clamp(x)) (= ~0 / ~1). The lut / linear_approx objects are built once per array (built per call they
// cost a stack round trip each).
using lut_t = aie::lut<4, float, bfloat16>;
using la_t = aie::linear_approx<bfloat16, lut_t>;
static inline __attribute__((always_inline)) B32 gelu_lut_i(B32 x, la_t &la) {
  B32 xc = aie::max(aie::min(x, aie::broadcast<bfloat16, 32>(SGELU_HI - 1.0f / 64)), aie::broadcast<bfloat16, 32>(SGELU_LO));
  A32 s = aie::concat(la.compute(xc.template extract<16>(0)), la.compute(xc.template extract<16>(1)));
  B32 sh = BF(s);
  B32 sl = BF(aie::mac(s, sh, (bfloat16)-1.0f));
  return BF(aie::mac(aie::mul(x, sh), x, sl));
}
static __attribute__((noinline)) void gelu_lut_n(const bfloat16 *__restrict in, bfloat16 *__restrict out, int n) {
  const lut_t lut(SGELU_N, sgelu_ab, sgelu_cd);
  la_t la(lut, SGELU_STEP, SGELU_BIAS);
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int i = 0; i < n; i += 64) {
    B32 a = gelu_lut_i(aie::load_v<32>(in + i), la), b = gelu_lut_i(aie::load_v<32>(in + i + 32), la);
    aie::store_v(out + i, a);
    aie::store_v(out + i + 32, b);
  }
}



#ifndef CH
#define CH 2048
#endif

template <typename F>
static inline void map(const bfloat16 *__restrict in, bfloat16 *__restrict out, F f) {
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int i = 0; i < CH; i += 64) {
    B32 a = f(aie::load_v<32>(in + i)), b = f(aie::load_v<32>(in + i + 32));
    aie::store_v(out + i, a);
    aie::store_v(out + i + 32, b);
  }
}

// The same with the lookups written out (mlir-aie aie_kernels/common/activations.h lut_segments_acc): the index is
// a byte offset, 16 B per segment in the doubled-group layout, so for segments of 2^-k: floor(x 2^(k+4)) + 16 bias.
template <int shift, int bias_bytes>
static inline __attribute__((always_inline)) A32 lut_lin32(const float *ab, const float *cd, B32 xc) {
  A32 offset;
  B32 slope;
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const aie::vector<int32, 16> index =
        aie::add(aie::vector<int32, 16>(bfloat16_to_int(xc.template extract<16>(h), shift)), bias_bytes);
    v32bfloat16 coeff0, coeff1;
    load_lut_2x_float(ab, cd, index, coeff0, coeff1);
    offset.insert(h, aie::accum<accfloat, 16>((v16accfloat)::shuffle(coeff0, coeff1, T32_16x2_hi)));
    slope.insert(h, aie::vector<bfloat16, 32>(::shuffle(coeff0, coeff1, T16_16x4_lo)).template extract<16>(1));
  }
  return aie::mac(offset, slope, xc);
}
constexpr int SGELU_SHIFT = -SGELU_STEP + 4, SGELU_BIAS_B = SGELU_BIAS * 16;
static inline __attribute__((always_inline)) B32 gelu_lut2_i(B32 x) {
  const B32 xc = aie::max(aie::min(x, aie::broadcast<bfloat16, 32>(SGELU_HI - 1.0f / 64)), aie::broadcast<bfloat16, 32>(SGELU_LO));
  A32 s = lut_lin32<SGELU_SHIFT, SGELU_BIAS_B>(sgelu_ab, sgelu_cd, xc);
  B32 sh = BF(s);
  B32 sl = BF(aie::mac(s, sh, (bfloat16)-1.0f));
  return BF(aie::mac(aie::mul(x, sh), x, sl));
}
static __attribute__((noinline)) void gelu_lut2_n(const bfloat16 *__restrict in, bfloat16 *__restrict out, int n) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int i = 0; i < n; i += 64) {
    B32 a = gelu_lut2_i(aie::load_v<32>(in + i)), b = gelu_lut2_i(aie::load_v<32>(in + i + 32));
    aie::store_v(out + i, a);
    aie::store_v(out + i + 32, b);
  }
}

extern "C" {
// mode: 0 copy, 1 vtanh (aie::tanh on the fp32 value), 2 nk8 gelu (current), 3 nk8 exp32, 4 aie::exp2 (fp32 in),
// 5 nk8 sigmoid, 6 aie::inv (bf16), 7 LUT gelu
void kprobe(bfloat16 *in, bfloat16 *out, int32_t mode) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const nk8::ExpC kc;
  const uint32_t t0 = cyc();
  for (int rep = 0; rep < REPS; rep++) {
  if (rep == 1) in = out;  // later repetitions map the previous result in place: no repetition can be elided
  if (mode == 0) {
    map(in, out, [](B32 x) { return x; });
  } else if (mode == 1) {
    map(in, out, [](B32 x) { return aie::tanh<bfloat16>(A32(x).template to_vector<float>()); });
  } else if (mode == 2) {
    map(in, out, [](B32 x) { return x; });  // (subtract mode 0's cycles)
    nk8::gelu_n(out, CH / 32, kc);
  } else if (mode == 3) {
    map(in, out, [&](B32 x) { return nk8::exp32_i(x, kc, aie::zeros<bfloat16, 32>()); });
  } else if (mode == 4) {
    map(in, out, [](B32 x) { return aie::exp2<bfloat16>(A32(x).template to_vector<float>()); });
  } else if (mode == 5) {
    map(in, out, [&](B32 x) { return BF(nk8::sigmoid32_i(x, kc)); });
  } else if (mode == 6) {
    map(in, out, [](B32 x) { return aie::inv(x); });
  } else if (mode == 7) {
    gelu_lut_n(in, out, CH);
  } else if (mode == 8) {
    gelu_lut2_n(in, out, CH);
  }
  }
  const uint32_t t1 = cyc();
  uint32_t *st = (uint32_t *)(out + CH);
  st[0] = t1 - t0;
  st[1] = (uint32_t)mode;
}
}
