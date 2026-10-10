// Loop-structure experiments for the SSM table helpers (round 9): compile with -S and count the hardware-loop
// bundles (build/asm/loopcount.py). Not linked anywhere.
#include <aie_api/aie.hpp>

#include "../nk8.h"
#include "ssm_lut.h"
#include "ssm9.h"

using B32 = aie::vector<bfloat16, 32>;

// V2: pointer bumps, constant trip count, two vectors per trip
template <int N>
__attribute__((noinline)) void sig_v2(bfloat16 *__restrict v) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(N / 2)
  AIE_LOOP_MAX_ITERATION_COUNT(N / 2)
  for (int i = 0; i < N / 2; i++) {
    const B32 a = ssm9::sig_lut_i(aie::load_v<32>(v)), b = ssm9::sig_lut_i(aie::load_v<32>(v + 32));
    aie::store_v(v, a);
    aie::store_v(v + 32, b);
    v += 64;
  }
}
// V3: one vector per trip, separate in / out pointers
template <int N>
__attribute__((noinline)) void sig_v3(const bfloat16 *__restrict in, bfloat16 *__restrict out) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(N)
  AIE_LOOP_MAX_ITERATION_COUNT(N)
  for (int i = 0; i < N; i++) {
    aie::store_v(out, ssm9::sig_lut_i(aie::load_v<32>(in)));
    in += 32;
    out += 32;
  }
}
// V4: one vector per trip, in place
template <int N>
__attribute__((noinline)) void sig_v4(bfloat16 *__restrict v) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(N)
  AIE_LOOP_MAX_ITERATION_COUNT(N)
  for (int i = 0; i < N; i++) {
    aie::store_v(v, ssm9::sig_lut_i(aie::load_v<32>(v)));
    v += 32;
  }
}
template <int N>
__attribute__((noinline)) void sqrt_v4(bfloat16 *__restrict v) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(N)
  AIE_LOOP_MAX_ITERATION_COUNT(N)
  for (int i = 0; i < N; i++) {
    aie::store_v(v, ssm9::sqrt_lut_i(aie::load_v<32>(v)));
    v += 32;
  }
}
// exp32a_i + a^2 + om, one vector per trip (same numerics as nk8::exp_a2_n)
template <int N>
__attribute__((noinline)) void expa2_v4(bfloat16 *__restrict v, bfloat16 *__restrict om, const nk8::ExpC &kc) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(N)
  AIE_LOOP_MAX_ITERATION_COUNT(N)
  for (int i = 0; i < N; i++) {
    nk8::A32 a = nk8::exp32a_i(aie::load_v<32>(v), kc);
    B32 ah, al;
    nk8::split(a, ah, al);
    B32 a2 = NK_BF(aie::mac(aie::mul(ah, ah), ah, NK_BF(aie::mul(al, (bfloat16)2.0f))));
    aie::store_v(v, NK_BF(a));
    aie::store_v(om, NK_BF(aie::msc(kc.c0, a2, kc.ones)));
    v += 32;
    om += 32;
  }
}

extern "C" void x9_all(bfloat16 *a, bfloat16 *b, bfloat16 *c) {
  const nk8::ExpC kc;
  sig_v2<32>(a);
  sig_v3<32>(a, b);
  sig_v4<32>(a);
  sqrt_v4<16>(a);
  expa2_v4<16>(a, c, kc);
}

// exp_a2 as two pipelined passes (the same numerics as nk8::exp32a_i + a^2 + om):
// A: la -> f (bf16) and n16 (int16), B: (f, n16) -> a, om
template <int N>
__attribute__((noinline)) void expA(const bfloat16 *__restrict la, bfloat16 *__restrict fo, int16_t *__restrict no, const nk8::ExpC &kc) {
  const bfloat16 L_HI = 1.4426950408889634f;
  const bfloat16 L_LO = 1.4426950408889634f - (float)L_HI;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(N)
  AIE_LOOP_MAX_ITERATION_COUNT(N)
  for (int i = 0; i < N; i++) {
    B32 d = aie::load_v<32>(la);
    d = aie::max(d, aie::broadcast<bfloat16, 32>(-60.0f));
    d = aie::min(d, aie::broadcast<bfloat16, 32>(60.0f));
    nk8::A32 za = aie::mul(d, L_HI);
    za = aie::mac(za, d, L_LO);
    nk8::A32 zn = aie::mac(za, kc.ones, (bfloat16)nk8::M0);
    B32 nb = NK_BF(aie::mac(zn, kc.ones, (bfloat16)(-nk8::M0)));
    aie::store_v(fo, NK_BF(aie::mac(za, nb, (bfloat16)-1.0f)));
    aie::vector<int16_t, 64> h = zn.template to_vector<float>().template cast_to<int16_t>();
    aie::store_v(no, aie::filter_even(h, 1));
    la += 32;
    fo += 32;
    no += 32;
  }
}
template <int N>
__attribute__((noinline)) void expB(const bfloat16 *__restrict fi, const int16_t *__restrict ni, bfloat16 *__restrict ao, bfloat16 *__restrict om, const nk8::ExpC &kc) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(N)
  AIE_LOOP_MAX_ITERATION_COUNT(N)
  for (int i = 0; i < N; i++) {
    const B32 f = aie::load_v<32>(fi);
    const nk8::I16 n16 = aie::load_v<32>(ni);
    const B32 p2n = aie::add(aie::upshift(n16, 7), aie::broadcast<int16_t, 32>(0x3F80)).template cast_to<bfloat16>();
    const B32 f2 = NK_BF(aie::mul(f, f));
    const B32 f3 = NK_BF(aie::mul(f2, f));
    nk8::A32 p = aie::mac(kc.c0, f, kc.c1);
    p = aie::mac(p, f2, kc.c2);
    p = aie::mac(p, f3, kc.c3);
    B32 ph, pl;
    nk8::split(p, ph, pl);
    nk8::A32 r = aie::mul(ph, p2n);
    r = aie::mac(r, pl, p2n);
    B32 ah, al;
    nk8::split(r, ah, al);
    const B32 a2 = NK_BF(aie::mac(aie::mul(ah, ah), ah, NK_BF(aie::mul(al, (bfloat16)2.0f))));
    aie::store_v(ao, ah);
    aie::store_v(om, NK_BF(aie::msc(kc.c0, a2, kc.ones)));
    fi += 32;
    ni += 32;
    ao += 32;
    om += 32;
  }
}
extern "C" void x9_exp(bfloat16 *la, bfloat16 *f, int16_t *n, bfloat16 *om, const nk8::ExpC &kc) {
  expA<16>(la, f, n, kc);
  expB<16>(f, n, la, om, kc);
}

template <int N>
__attribute__((noinline)) void expB1(const bfloat16 *__restrict fi, bfloat16 *__restrict pho, bfloat16 *__restrict plo, const nk8::ExpC &kc) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(N)
  AIE_LOOP_MAX_ITERATION_COUNT(N)
  for (int i = 0; i < N; i++) {
    const B32 f = aie::load_v<32>(fi);
    const B32 f2 = NK_BF(aie::mul(f, f));
    const B32 f3 = NK_BF(aie::mul(f2, f));
    nk8::A32 p = aie::mac(kc.c0, f, kc.c1);
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
template <int N>
__attribute__((noinline)) void expB2(const bfloat16 *__restrict phi, const bfloat16 *__restrict pli, const int16_t *__restrict ni, bfloat16 *__restrict ao, bfloat16 *__restrict om, const nk8::ExpC &kc) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(N)
  AIE_LOOP_MAX_ITERATION_COUNT(N)
  for (int i = 0; i < N; i++) {
    const B32 ph = aie::load_v<32>(phi), pl = aie::load_v<32>(pli);
    const nk8::I16 n16 = aie::load_v<32>(ni);
    const B32 p2n = aie::add(aie::upshift(n16, 7), aie::broadcast<int16_t, 32>(0x3F80)).template cast_to<bfloat16>();
    nk8::A32 r = aie::mul(ph, p2n);
    r = aie::mac(r, pl, p2n);
    B32 ah, al;
    nk8::split(r, ah, al);
    const B32 a2 = NK_BF(aie::mac(aie::mul(ah, ah), ah, NK_BF(aie::mul(al, (bfloat16)2.0f))));
    aie::store_v(ao, ah);
    aie::store_v(om, NK_BF(aie::msc(kc.c0, a2, kc.ones)));
    phi += 32;
    pli += 32;
    ni += 32;
    ao += 32;
    om += 32;
  }
}
extern "C" void x9_exp3(bfloat16 *la, bfloat16 *f, int16_t *n, bfloat16 *om, bfloat16 *ph, bfloat16 *pl, const nk8::ExpC &kc) {
  expA<16>(la, f, n, kc);
  expB1<16>(f, ph, pl, kc);
  expB2<16>(ph, pl, n, la, om, kc);
}

template <int N>
__attribute__((noinline)) void expC1(const bfloat16 *__restrict phi, const bfloat16 *__restrict pli, const int16_t *__restrict ni, bfloat16 *__restrict ao, bfloat16 *__restrict alo) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(N)
  AIE_LOOP_MAX_ITERATION_COUNT(N)
  for (int i = 0; i < N; i++) {
    const B32 ph = aie::load_v<32>(phi), pl = aie::load_v<32>(pli);
    const nk8::I16 n16 = aie::load_v<32>(ni);
    const B32 p2n = aie::add(aie::upshift(n16, 7), aie::broadcast<int16_t, 32>(0x3F80)).template cast_to<bfloat16>();
    nk8::A32 r = aie::mul(ph, p2n);
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
template <int N>
__attribute__((noinline)) void expC2(const bfloat16 *__restrict ahi, const bfloat16 *__restrict ali, bfloat16 *__restrict om, const nk8::ExpC &kc) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(N)
  AIE_LOOP_MAX_ITERATION_COUNT(N)
  for (int i = 0; i < N; i++) {
    const B32 ah = aie::load_v<32>(ahi), al = aie::load_v<32>(ali);
    const B32 a2 = NK_BF(aie::mac(aie::mul(ah, ah), ah, NK_BF(aie::mul(al, (bfloat16)2.0f))));
    aie::store_v(om, NK_BF(aie::msc(kc.c0, a2, kc.ones)));
    ahi += 32;
    ali += 32;
    om += 32;
  }
}
extern "C" void x9_exp4(bfloat16 *la, bfloat16 *f, int16_t *n, bfloat16 *om, bfloat16 *ph, bfloat16 *pl, const nk8::ExpC &kc) {
  expA<16>(la, f, n, kc);
  expB1<16>(f, ph, pl, kc);
  expC1<16>(ph, pl, n, la, f);
  expC2<16>(la, f, om, kc);
}
