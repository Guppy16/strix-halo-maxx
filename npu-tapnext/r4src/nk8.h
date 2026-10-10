// Shared bf16 numerics for design8's kernels: exact-enough exp (attention v15's exp32 with a hi/lo argument),
// a reciprocal with two Newton corrections carried in fp32, sigmoid, GELU. All on 32 bf16 lanes; fp32 only inside
// the MAC accumulator (fp32 vector arithmetic is emulated on AIE2P).
#pragma once
#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

namespace nk8 {
using B32 = aie::vector<bfloat16, 32>;
using A32 = aie::accum<accfloat, 32>;
using I16 = aie::vector<int16_t, 32>;
#define NK_BF(e) ((e).template to_vector<bfloat16>())

struct ExpC {
  A32 c0;
  B32 ones, c1, c2, c3;
  ExpC() : c0(aie::broadcast<float, 32>(1.0f)), ones(aie::broadcast<bfloat16, 32>(1.0f)),
           c1(aie::broadcast<bfloat16, 32>(0.69314718f)), c2(aie::broadcast<bfloat16, 32>(0.24022651f)),
           c3(aie::broadcast<bfloat16, 32>(0.05550411f)) {}
};
constexpr float M0 = 12582912.0f;  // 1.5 * 2^23: fp32 ulp 1, so z + M0 is round(z) + M0 with n in the low mantissa bits

static inline void split(const A32 &a, B32 &hi, B32 &lo) {
  hi = NK_BF(a);
  lo = NK_BF(aie::mac(a, hi, (bfloat16)-1.0f));
}

// e = bf(exp(d + dl)), |d| <= 60: 2^(z) with z = d log2e in fp32, n = round(z) by the magic add, cubic 2^f
static inline __attribute__((always_inline)) B32 exp32_i(B32 d, const ExpC &kc, B32 dl) {
  const bfloat16 L_HI = 1.4426950408889634f;
  const bfloat16 L_LO = 1.4426950408889634f - (float)L_HI;
  d = aie::max(d, aie::broadcast<bfloat16, 32>(-60.0f));
  d = aie::min(d, aie::broadcast<bfloat16, 32>(60.0f));
  A32 za = aie::mul(d, L_HI);
  za = aie::mac(za, d, L_LO);
  za = aie::mac(za, dl, L_HI);
  A32 zn = aie::mac(za, kc.ones, (bfloat16)M0);
  B32 nb = NK_BF(aie::mac(zn, kc.ones, (bfloat16)(-M0)));
  B32 f = NK_BF(aie::mac(za, nb, (bfloat16)-1.0f));
  aie::vector<int16_t, 64> h = zn.template to_vector<float>().template cast_to<int16_t>();
  I16 n16 = aie::filter_even(h, 1);
  B32 p2n = aie::add(aie::upshift(n16, 7), aie::broadcast<int16_t, 32>(0x3F80)).template cast_to<bfloat16>();
  B32 f2 = NK_BF(aie::mul(f, f));
  B32 f3 = NK_BF(aie::mul(f2, f));
  A32 p = aie::mac(kc.c0, f, kc.c1);
  p = aie::mac(p, f2, kc.c2);
  p = aie::mac(p, f3, kc.c3);
  return NK_BF(aie::mul(NK_BF(p), p2n));
}

// 1 / (1 + e) as an fp32 accumulator: int16 bit-trick seed (|err| < 5.3%), two Newton corrections carried in fp32.
// (aie::inv is the scalar elementary unit: 32 extracts + inverses per vector, which made every sigmoid/GELU slow)
static inline __attribute__((always_inline)) A32 recip1p_i(B32 e, const ExpC &kc) {
  A32 d = aie::mac(kc.c0, e, kc.ones);
  B32 dh, dl;
  split(d, dh, dl);
  B32 r0 = aie::sub(aie::broadcast<int16_t, 32>((int16_t)0x7EF3), dh.template cast_to<int16_t>()).template cast_to<bfloat16>();
  A32 eps = aie::msc(kc.c0, dh, r0);
  eps = aie::msc(eps, dl, r0);
  A32 r = aie::mac(A32(aie::mul(r0, kc.ones)), r0, NK_BF(eps));
  B32 rh, rl;  // second correction (seed err 5.3% -> 2.8e-3 -> 8e-6, below bf16's 2^-9: no third needed)
  split(r, rh, rl);
  A32 e2 = aie::msc(kc.c0, dh, rh);
  e2 = aie::msc(e2, dh, rl);
  e2 = aie::msc(e2, dl, rh);
  return aie::mac(r, rh, NK_BF(e2));
}

// sigmoid(z) in fp32 (round once at the use site)
static inline __attribute__((always_inline)) A32 sigmoid32_i(B32 z, const ExpC &kc) { return recip1p_i(exp32_i(NK_BF(aie::mul(z, (bfloat16)-1.0f)), kc, aie::zeros<bfloat16, 32>()), kc); }

// gelu_tanh(x) = x * sigmoid(2u), u = x (c0 + c1 x^2) carried in fp32; the product as an fp32 accumulator
// (the clamp to [-8, 8] only bounds the exp argument: the product uses the unclamped x, so gelu(x) = x for large x)
static inline __attribute__((always_inline)) A32 gelu32_i(B32 x_in, const ExpC &kc) {
  B32 x = aie::max(x_in, aie::broadcast<bfloat16, 32>(-8.0f));
  x = aie::min(x, aie::broadcast<bfloat16, 32>(8.0f));
  const A32 c0a(aie::broadcast<float, 32>(0.7978845608f));
  const bfloat16 C1H = 0.035677408f, C1L = 0.035677408f - (float)(bfloat16)0.035677408f;
  A32 ta = aie::mul(x, x);
  B32 th, tl;
  split(ta, th, tl);
  A32 wa = aie::mac(c0a, th, C1H);
  wa = aie::mac(wa, th, C1L);
  wa = aie::mac(wa, tl, C1H);
  B32 wh, wl;
  split(wa, wh, wl);
  A32 ua = aie::mul(x, wh);
  ua = aie::mac(ua, x, wl);
  B32 uh, ul;
  split(ua, uh, ul);
  B32 e = exp32_i(NK_BF(aie::mul(uh, (bfloat16)-2.0f)), kc, NK_BF(aie::mul(ul, (bfloat16)-2.0f)));
  A32 s = recip1p_i(e, kc);
  B32 sh, sl;
  split(s, sh, sl);
  A32 g = aie::mul(x_in, sh);
  return aie::mac(g, x_in, sl);
}

// sqrt(w) for bf16 w >= 0 (w = 0 -> 0): int16 bit-trick rsqrt seed, two Newton steps, times w
static inline __attribute__((always_inline)) B32 sqrt32_i(B32 w, const ExpC &kc) {
  B32 wc = aie::max(w, aie::broadcast<bfloat16, 32>(1e-30f));
  I16 bits = wc.template cast_to<int16_t>();
  B32 r = aie::sub(aie::broadcast<int16_t, 32>((int16_t)0x5F37), aie::downshift(bits, 1)).template cast_to<bfloat16>();
  const A32 c15(aie::broadcast<float, 32>(1.5f));
  B32 hw = NK_BF(aie::mul(wc, (bfloat16)0.5f));
  for (int i = 0; i < 2; i++) {
    B32 rr = NK_BF(aie::mul(r, r));
    B32 t = NK_BF(aie::msc(c15, hw, rr));
    r = NK_BF(aie::mul(r, t));
  }
  return NK_BF(aie::mul(w, r));  // w = 0 gives 0
}
// out-of-line wrappers (one copy per object) and array forms: the array forms loop over n 32-lane vectors in place
// so the compiler can overlap independent iterations (the math is one long dependency chain per vector)
static __attribute__((noinline, minsize)) B32 exp32(B32 d, const ExpC &kc, B32 dl = aie::zeros<bfloat16, 32>()) { return exp32_i(d, kc, dl); }
static __attribute__((noinline, minsize)) A32 recip1p(B32 e, const ExpC &kc) { return recip1p_i(e, kc); }
static __attribute__((noinline, minsize)) A32 sigmoid32(B32 z, const ExpC &kc) { return sigmoid32_i(z, kc); }
static __attribute__((noinline, minsize)) A32 gelu32(B32 x, const ExpC &kc) { return gelu32_i(x, kc); }
static __attribute__((noinline, minsize)) B32 sqrt32(B32 w, const ExpC &kc) { return sqrt32_i(w, kc); }

// Two independent vectors per call / iteration: Peano does not software-pipeline these long dependency chains, but
// it interleaves two chains in one basic block (measured loop bundles per vector: exp 73 -> 38, sigmoid 140 -> 77,
// gelu 199 -> 111; four chains spill and gain nothing). All n are even.
static __attribute__((noinline)) void exp32x2(B32 &a, B32 &b, const ExpC &kc) {
  B32 ea = exp32_i(a, kc, aie::zeros<bfloat16, 32>()), eb = exp32_i(b, kc, aie::zeros<bfloat16, 32>());
  a = ea;
  b = eb;
}
static __attribute__((noinline)) void gelu32x2(B32 &a, B32 &b, const ExpC &kc) {  // bf16(gelu) in place
  B32 ga = NK_BF(gelu32_i(a, kc)), gb = NK_BF(gelu32_i(b, kc));
  a = ga;
  b = gb;
}
static __attribute__((noinline)) void exp_n(bfloat16 *v, int n, const ExpC &kc) {
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (int i = 0; i < n; i += 2) {
    B32 a = exp32_i(aie::load_v<32>(v + 32 * i), kc, aie::zeros<bfloat16, 32>());
    B32 b = exp32_i(aie::load_v<32>(v + 32 * i + 32), kc, aie::zeros<bfloat16, 32>());
    aie::store_v(v + 32 * i, a);
    aie::store_v(v + 32 * i + 32, b);
  }
}
static __attribute__((noinline)) void sigmoid_n(bfloat16 *v, int n, const ExpC &kc) {
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (int i = 0; i < n; i += 2) {
    B32 a = NK_BF(sigmoid32_i(aie::load_v<32>(v + 32 * i), kc));
    B32 b = NK_BF(sigmoid32_i(aie::load_v<32>(v + 32 * i + 32), kc));
    aie::store_v(v + 32 * i, a);
    aie::store_v(v + 32 * i + 32, b);
  }
}
static __attribute__((noinline)) void sqrt_n(bfloat16 *v, int n, const ExpC &kc) {
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (int i = 0; i < n; i += 2) {
    B32 a = sqrt32_i(aie::load_v<32>(v + 32 * i), kc);
    B32 b = sqrt32_i(aie::load_v<32>(v + 32 * i + 32), kc);
    aie::store_v(v + 32 * i, a);
    aie::store_v(v + 32 * i + 32, b);
  }
}
}  // namespace nk8
