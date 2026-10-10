// Shared bf16 exp / reciprocal built from native MAC-unit ops (see mm_f32acc.cc -DGELU_BF16).
#pragma once
// hi/lo split of an fp32 accumulator into two bf16 vectors (hi + lo ~ 16 significant bits), MAC-unit only
static inline void split_acc(const A32 &a, B32 &hi, B32 &lo) {
  hi = BF(a);
  lo = BF(aie::mac(a, hi, bfloat16(-1.0f)));
}
// p = c + f * p with f and p as hi/lo pairs (3 MACs): keeps ~2^-16 precision through a Horner chain
static inline A32 fma_split(float c, B32 fh, B32 fl, B32 ph, B32 pl) {
  A32 a = aie::accum<accfloat, 32>(aie::broadcast<float, 32>(c));
  a = aie::mac(a, fh, ph);
  a = aie::mac(a, fh, pl);
  a = aie::mac(a, fl, ph);
  return a;
}
// exp(x) for x given as an fp32 accumulator (|x| <= 60): 2^(x log2 e), integer part into the exponent bits,
// 2^f by a degree-4 split-operand Horner. Result rounded to bf16: matches a correctly rounded exp except
// when the true value lies within ~1e-5 of a rounding boundary.
static inline B32 exp_acc(A32 xa) {
  B32 xh, xl;
  split_acc(xa, xh, xl);
  const bfloat16 L_HI = 1.4426950408889634f;
  const bfloat16 L_LO = 1.4426950408889634f - (float)L_HI;
  A32 za = aie::mul(xh, L_HI);
  za = aie::mac(za, xh, L_LO);
  za = aie::mac(za, xl, L_HI);
  aie::vector<float, 32> z = za.template to_vector<float>();
  z = aie::max(aie::min(z, aie::broadcast<float, 32>(100.0f)), aie::broadcast<float, 32>(-100.0f));
  aie::vector<int32_t, 32> n = aie::to_fixed<int32_t>(aie::sub(z, 0.5f), 0);  // floor
  A32 fa = aie::accum<accfloat, 32>(aie::sub(z, aie::to_float(n, 0)));        // f in [0, 1)
  B32 fh, fl, ph, pl;
  split_acc(fa, fh, fl);
  A32 p = aie::accum<accfloat, 32>(aie::broadcast<float, 32>(0.009618129f));
  split_acc(p, ph, pl);
  p = fma_split(0.05550411f, fh, fl, ph, pl); split_acc(p, ph, pl);
  p = fma_split(0.2402265f, fh, fl, ph, pl); split_acc(p, ph, pl);
  p = fma_split(0.6931472f, fh, fl, ph, pl); split_acc(p, ph, pl);
  p = fma_split(1.0f, fh, fl, ph, pl);
  aie::vector<int32_t, 32> ebits = aie::add(aie::upshift(n, 23), p.template to_vector<float>().template cast_to<int32_t>());
  return aie::accum<accfloat, 32>(ebits.template cast_to<float>()).template to_vector<bfloat16>();
}
// 1/d for a bf16 d > 0, correctly rounded to bf16 in all but boundary cases: bit-trick seed, then two
// Newton steps carried in the fp32 accumulator with split operands.
static inline B32 recip_b(B32 d) {
  aie::vector<int16_t, 32> ib = aie::sub(aie::broadcast<int16_t, 32>((int16_t)0x7EF3), d.template cast_to<int16_t>());
  B32 y = ib.template cast_to<bfloat16>();
  B32 yh = y, yl = aie::zeros<bfloat16, 32>();
  for (int i = 0; i < 2; i++) {
    // r = 2 - d * y  (fp32), y = y * r (fp32)
    A32 r = aie::accum<accfloat, 32>(aie::broadcast<float, 32>(2.0f));
    r = aie::mac(r, d, aie::neg(yh));
    r = aie::mac(r, d, aie::neg(yl));
    B32 rh, rl;
    split_acc(r, rh, rl);
    A32 yn = aie::mul(yh, rh);
    yn = aie::mac(yn, yh, rl);
    yn = aie::mac(yn, yl, rh);
    split_acc(yn, yh, yl);
  }
  return yh;
}

