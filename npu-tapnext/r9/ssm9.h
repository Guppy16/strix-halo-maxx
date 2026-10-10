// RG-LRU transcendentals by table lookup (round 9; numerics chosen on real activations with r9/ssmemu.py, tables from
// r9/ssmlut.py -> ssm_lut.h, which the includer includes first with SSM_LUT_AB / SSM_LUT_CD set to two banks).
//   sig_lut_i   bf16(sigmoid(x)): line per 1/8 segment on [-12, 8), x clamped (5% bf16 mismatches vs torch; nk8 19%)
//   a_om_lut_i  a = bf16(exp(la)), om = bf16(1 - bf16(exp(2 la))): 2^n 2^f with 2^f from a 64-segment table on the
//               hi/lo f; e2 from p^2 (p hi/lo) 2^2n (a 1.3% / mult 1.0% mismatches; nk8 13% / 40%)
//   sqrt_lut_i  exact RNE sqrt of a bf16 w >= 0 (0 -> 0): mantissa table (exponent parity, 7 bits), exponent halved
// Each 16-lane lookup is 4 loads on the ldb slot (the table's two copies must sit in different banks).
#pragma once
#include <aie_api/aie.hpp>

#include "../nk8.h"

namespace ssm9 {
using nk8::A32;
using nk8::B32;
using nk8::I16;
#define S9_BF(e) ((e).template to_vector<bfloat16>())

// the 16-lane (offset fp32, slope bf16) lookup of a float table, as mm8.cc gelu_lut_i
template <int SHIFT, int BIAS_B>
static inline __attribute__((always_inline)) void lut_line(const float *ab, const float *cd, B32 xc, A32 &off, B32 &slope) {
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const aie::vector<int32, 16> index =
        aie::add(aie::vector<int32, 16>(bfloat16_to_int(xc.template extract<16>(h), SHIFT)), BIAS_B);
    v32bfloat16 coeff0, coeff1;
    load_lut_2x_float(ab, cd, index, coeff0, coeff1);
    off.insert(h, aie::accum<accfloat, 16>((v16accfloat)::shuffle(coeff0, coeff1, T32_16x2_hi)));
    slope.insert(h, aie::vector<bfloat16, 32>(::shuffle(coeff0, coeff1, T16_16x4_lo)).template extract<16>(1));
  }
}

static inline __attribute__((always_inline)) B32 sig_lut_i(B32 x) {
  constexpr float XHI = SSIG_HI - 0.5f / (1 << -SSIG_STEP);  // inside the last segment, exact in bf16
  const B32 xc = aie::max(aie::min(x, aie::broadcast<bfloat16, 32>(XHI)), aie::broadcast<bfloat16, 32>(SSIG_LO));
  A32 off;
  B32 slope;
  lut_line<-SSIG_STEP + 4, SSIG_BIAS * 16>(ssig_ab, ssig_cd, xc, off, slope);
  return S9_BF(aie::mac(off, slope, xc));
}

#ifdef SSM_LUT_EXP
static inline __attribute__((always_inline)) void a_om_lut_i(B32 la, B32 &a, B32 &om, const nk8::ExpC &kc) {
  const bfloat16 L_HI = 1.4426950408889634f;
  const bfloat16 L_LO = 1.4426950408889634f - (float)L_HI;
  A32 za = aie::mul(la, L_HI);
  za = aie::mac(za, la, L_LO);
  const A32 zn = aie::mac(za, kc.ones, (bfloat16)nk8::M0);  // n = round(z) in the low mantissa bits
  const B32 nb = S9_BF(aie::mac(zn, kc.ones, (bfloat16)(-nk8::M0)));
  const A32 fa = aie::mac(za, nb, (bfloat16)-1.0f);  // f = z - n, |f| <= 1/2
  const B32 fh = S9_BF(fa);
  const B32 fl = S9_BF(aie::mac(fa, fh, (bfloat16)-1.0f));
  const I16 n16 = aie::filter_even(zn.template to_vector<float>().template cast_to<int16_t>(), 1);
  constexpr float FHI = 0.5f - 0.5f / SEXP_N;
  const B32 fc = aie::max(aie::min(fh, aie::broadcast<bfloat16, 32>(FHI)), aie::broadcast<bfloat16, 32>(-0.5f));
  A32 off;
  B32 slope;
  lut_line<-SEXP_STEP + 4, SEXP_BIAS * 16>(sexp_ab, sexp_cd, fc, off, slope);
  A32 p = aie::mac(off, slope, fh);
  p = aie::mac(p, slope, fl);
  const B32 ph = S9_BF(p);
  const B32 pl = S9_BF(aie::mac(p, ph, (bfloat16)-1.0f));
  a = aie::add(ph.template cast_to<int16_t>(), aie::upshift(n16, 7)).template cast_to<bfloat16>();
  const B32 p2 = S9_BF(aie::mac(aie::mul(ph, ph), ph, S9_BF(aie::mul(pl, (bfloat16)2.0f))));
  const B32 e2 = aie::add(p2.template cast_to<int16_t>(), aie::upshift(n16, 8)).template cast_to<bfloat16>();
  om = S9_BF(aie::msc(kc.c0, e2, kc.ones));
}
#endif

static inline __attribute__((always_inline)) B32 sqrt_lut_i(B32 w) {
  const I16 b = w.template cast_to<int16_t>();
  const I16 idx = aie::bit_and(b, aie::broadcast<int16_t, 32>((int16_t)0xFF));
  aie::accum<acc32, 32> ia;
  ia.from_vector(idx, 2);  // 4 B per doubled entry
  v64int8 c0, c1, c2, c3;
  ::load_lut_2x_int8(ssqrt_ab, ssqrt_cd, ia.template extract<16>(0).template to_vector<int32>(0), c0, c1);
  ::load_lut_2x_int8(ssqrt_ab, ssqrt_cd, ia.template extract<16>(1).template to_vector<int32>(0), c2, c3);
  c0 = ::shuffle(c0, c1, T16_16x4_lo);
  c2 = ::shuffle(c2, c3, T16_16x4_lo);
  const I16 t = I16((v32int16)::shuffle(c0, c2, T256_2x2_lo));
  const I16 k = aie::downshift(aie::sub(aie::downshift(b, 7), aie::broadcast<int16_t, 32>((int16_t)127)), 1);
  const B32 r = aie::add(t, aie::upshift(k, 7)).template cast_to<bfloat16>();
  return aie::min(r, S9_BF(aie::mul(w, (bfloat16)1.152921504606846976e18f)));  // w = 0 -> 0 (2^60 w >= 1 >= r else)
}

// array forms, two chains per iteration (Peano interleaves two independent chains in one basic block)
static __attribute__((noinline)) void sigmoid_lut_n(bfloat16 *__restrict v, int n) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (int i = 0; i < 32 * n; i += 64) {
    const B32 a = sig_lut_i(aie::load_v<32>(v + i)), b = sig_lut_i(aie::load_v<32>(v + i + 32));
    aie::store_v(v + i, a);
    aie::store_v(v + i + 32, b);
  }
}
#ifdef SSM_LUT_EXP
// la -> a in place, om -> om[]
static __attribute__((noinline)) void a_om_lut_n(bfloat16 *__restrict v, bfloat16 *__restrict om, int n, const nk8::ExpC &kc) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (int i = 0; i < 32 * n; i += 64) {
    B32 a0, o0, a1, o1;
    a_om_lut_i(aie::load_v<32>(v + i), a0, o0, kc);
    a_om_lut_i(aie::load_v<32>(v + i + 32), a1, o1, kc);
    aie::store_v(v + i, a0);
    aie::store_v(v + i + 32, a1);
    aie::store_v(om + i, o0);
    aie::store_v(om + i + 32, o1);
  }
}
#endif
static __attribute__((noinline)) void sqrt_lut_n(bfloat16 *__restrict v, int n) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (int i = 0; i < 32 * n; i += 64) {
    const B32 a = sqrt_lut_i(aie::load_v<32>(v + i)), b = sqrt_lut_i(aie::load_v<32>(v + i + 32));
    aie::store_v(v + i, a);
    aie::store_v(v + i + 32, b);
  }
}
}  // namespace ssm9

// ---- exp_a2 as pipelined passes (round 9, SSM_EXP3): nk8::exp_a2_n's exact op sequence split into five loops that
// Peano software-pipelines (one long chain per vector was 63.5 bundles per 32 lanes with two chains interleaved; the
// passes compile to 11.5 + 20 + 12 + 6 + 4 bundles per vector). Bit-identical. Scratch: four 1 KB slots s1..s4.
namespace ssm9 {
// log_a = bf16(bf16(-8 gate_a) sp), in place; sp per column block (two vectors each). (Fused into pass A the loop did
// not pipeline: 50 bundles per vector against 11.5 + 20.)
static __attribute__((noinline)) void loga_p(bfloat16 *__restrict ga, const B32 *__restrict tsp) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  AIE_LOOP_MAX_ITERATION_COUNT(8)
  for (int cb = 0; cb < 8; cb++) {
    const B32 sp = *tsp++;
    const B32 x0 = aie::load_v<32>(ga), x1 = aie::load_v<32>(ga + 32);
    aie::store_v(ga, S9_BF(aie::mul(S9_BF(aie::mul(x0, (bfloat16)-8.0f)), sp)));
    aie::store_v(ga + 32, S9_BF(aie::mul(S9_BF(aie::mul(x1, (bfloat16)-8.0f)), sp)));
    ga += 64;
  }
}
// A: la -> f = bf16(z - round(z)) and n = round(z), z = la log2e
static __attribute__((noinline)) void exp_pA(const bfloat16 *__restrict la, bfloat16 *__restrict fo, int16_t *__restrict no) {
  const nk8::ExpC kc;  // local: through a reference Peano reloads the constants inside the loop (it cannot rule out aliasing)
  const bfloat16 L_HI = 1.4426950408889634f;
  const bfloat16 L_LO = 1.4426950408889634f - (float)L_HI;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  AIE_LOOP_MAX_ITERATION_COUNT(16)
  for (int i = 0; i < 16; i++) {
    B32 d = aie::load_v<32>(la);
    d = aie::max(d, aie::broadcast<bfloat16, 32>(-60.0f));
    d = aie::min(d, aie::broadcast<bfloat16, 32>(60.0f));
    A32 za = aie::mul(d, L_HI);
    za = aie::mac(za, d, L_LO);
    const A32 zn = aie::mac(za, kc.ones, (bfloat16)nk8::M0);
    const B32 nb = S9_BF(aie::mac(zn, kc.ones, (bfloat16)(-nk8::M0)));
    aie::store_v(fo, S9_BF(aie::mac(za, nb, (bfloat16)-1.0f)));
    const aie::vector<int16_t, 64> h = zn.template to_vector<float>().template cast_to<int16_t>();
    aie::store_v(no, aie::filter_even(h, 1));
    la += 32;
    fo += 32;
    no += 32;
  }
}
// B: f -> p = 1 + c1 f + c2 f^2 + c3 f^3 as hi / lo bf16
static __attribute__((noinline)) void exp_pB(const bfloat16 *__restrict fi, bfloat16 *__restrict pho, bfloat16 *__restrict plo) {
  const nk8::ExpC kc;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  AIE_LOOP_MAX_ITERATION_COUNT(16)
  for (int i = 0; i < 16; i++) {
    const B32 f = aie::load_v<32>(fi);
    const B32 f2 = S9_BF(aie::mul(f, f));
    const B32 f3 = S9_BF(aie::mul(f2, f));
    A32 p = aie::mac(kc.c0, f, kc.c1);
    p = aie::mac(p, f2, kc.c2);
    p = aie::mac(p, f3, kc.c3);
    B32 ph, pl;
    nk8::split(p, ph, pl);
    aie::store_v(pho, ph);
    aie::store_v(plo, pl);
    fi += 32;
    pho += 32;
    plo += 32;
  }
}
// C: a = p 2^n as hi (-> a) / lo
static __attribute__((noinline)) void exp_pC(const bfloat16 *__restrict phi, const bfloat16 *__restrict pli, const int16_t *__restrict ni,
                                             bfloat16 *__restrict ao, bfloat16 *__restrict alo) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  AIE_LOOP_MAX_ITERATION_COUNT(16)
  for (int i = 0; i < 16; i++) {
    const B32 ph = aie::load_v<32>(phi), pl = aie::load_v<32>(pli);
    const I16 n16 = aie::load_v<32>(ni);
    const B32 p2n = aie::add(aie::upshift(n16, 7), aie::broadcast<int16_t, 32>(0x3F80)).template cast_to<bfloat16>();
    A32 r = aie::mul(ph, p2n);
    r = aie::mac(r, pl, p2n);
    B32 ah, al;
    nk8::split(r, ah, al);
    aie::store_v(ao, ah);
    aie::store_v(alo, al);
    phi += 32;
    pli += 32;
    ni += 32;
    ao += 32;
    alo += 32;
  }
}
// D: om = bf16(1 - bf16(a^2)), a^2 from the hi / lo a
static __attribute__((noinline)) void exp_pD(const bfloat16 *__restrict ahi, const bfloat16 *__restrict ali, bfloat16 *__restrict om) {
  const nk8::ExpC kc;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  AIE_LOOP_MAX_ITERATION_COUNT(16)
  for (int i = 0; i < 16; i++) {
    const B32 ah = aie::load_v<32>(ahi), al = aie::load_v<32>(ali);
    const B32 a2 = S9_BF(aie::mac(aie::mul(ah, ah), ah, S9_BF(aie::mul(al, (bfloat16)2.0f))));
    aie::store_v(om, S9_BF(aie::msc(kc.c0, a2, kc.ones)));
    ahi += 32;
    ali += 32;
    om += 32;
  }
}
// gate_a [16 x 32] -> a in place and om -> s2: log_a, then the passes through the scratch slots
static inline void loga_exp_a2(bfloat16 *ga, const B32 *tsp, bfloat16 *s1, bfloat16 *s2, bfloat16 *s3, bfloat16 *s4) {
  loga_p(ga, tsp);
  exp_pA(ga, s1, (int16_t *)s2);
  exp_pB(s1, s3, s4);
  exp_pC(s3, s4, (const int16_t *)s2, ga, s1);
  exp_pD(ga, s1, s2);
}
}  // namespace ssm9
