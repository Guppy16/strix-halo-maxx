// round 16: one-core bench of r16/gen_ggk.py's ggk (both SSM gate GEMMs in one asm loop) against two calls of ew8.cc's
// gate_gemm (verbatim), random bfp operands and bf16 biases; in-situ banks (B bank 1, A copy / biases bank 2, out bank 1)
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "aie_kernel_utils.h"
using B64 = aie::vector<bfloat16, 64>;
using A64 = aie::accum<accfloat, 64>;
using MM = aie::mmul<8, 8, 8, bfp16ebs8>;
using BP = aie::block_vector<bfp16ebs8, 64>;
template <aie_dm_resource R>
using InS = aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, 64, R>;
#define BF(e) ((e).template to_vector<bfloat16>())
extern "C" void ggk(const bfp16ebs8 *xcp, const bfp16ebs8 *wt, const void *tbx, bfloat16 *dst, const void *tba);
static __attribute__((noinline)) void gate_gemm(const bfp16ebs8 *apan, const bfp16ebs8 *wt, const B64 *__restrict btab,
                                                bfloat16 *__restrict dst) {
  const B64 ones64 = aie::broadcast<bfloat16, 64>(1.0f);
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int np = 0; np < 4; np++) {
    MM acc0, acc1;
    InS<aie_dm_resource::a> as(apan);
    InS<aie_dm_resource::b> bs(wt + np * 1152);
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
      BP a = as.pop();
      BP b0 = bs.pop(), b1 = bs.pop();
      if (kk == 0) {
        acc0.mul(a, aie::op_transpose(b0));
        acc1.mul(a, aie::op_transpose(b1));
      } else {
        acc0.mac(a, aie::op_transpose(b0));
        acc1.mac(a, aie::op_transpose(b1));
      }
    }
    const A64 r0 = aie::mul(BF(acc0.to_accum()), ones64), r1 = aie::mul(BF(acc1.to_accum()), ones64);
    aie::store_v(dst + np * 128, BF(aie::mac(r0, btab[2 * np], ones64)));
    aie::store_v(dst + np * 128 + 64, BF(aie::mac(r1, btab[2 * np + 1], ones64)));
  }
}
#define SEC(b) __attribute__((section(".aie.bank" #b)))
SEC(1) alignas(64) static uint8_t g_w[9216 + 512];          // wx | wa (+ the drain's over-read)
SEC(2) alignas(64) static uint8_t g_a[576 + 64];
SEC(2) alignas(64) static B64 g_tb2[18];  // tbx | tba contiguous (BIAS_1D debug builds need it)
static B64 *const g_tbx = g_tb2, *const g_tba = g_tb2 + 8;
constexpr int GUARD = 256;  // bf16 elements each side
SEC(1) alignas(64) static bfloat16 g_out[1024 + 2 * GUARD];
alignas(64) static bfloat16 g_ref[1024];
static __attribute__((noinline)) uint32_t gstamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}
static uint32_t g_seed;
static uint32_t rnd() { g_seed = g_seed * 1664525u + 1013904223u; return g_seed >> 8; }
static void fill_bfp(uint8_t *p, int bytes, int e0, int er) {  // [E][8 x int8] groups of 9 bytes
  for (int i = 0; i < bytes; i++) p[i] = (i % 9 == 0) ? (uint8_t)(e0 + rnd() % er) : (uint8_t)rnd();
}
// x: [1] seed, [2] exponent base; y: [0] ggk cycles, [1] bad, [2] C++ cycles, [3] first bad, [4] got, [5] want, [6] guard
extern "C" void apb(int32_t *x, int32_t *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  g_seed = 11u + x[1];
  fill_bfp(g_w, 9216, x[2], 6);
  fill_bfp(g_a, 576, x[2], 6);
  if (x[3] > 0)  // debug: keep only panel x[3] - 1 of A (mantissas of the others zero)
    for (int i = 0; i < 576; i++) if (i / 72 != x[3] - 1 && i % 9) g_a[i] = 0;
  for (int t = 0; t < 2; t++) {
    B64 *tb = t ? g_tba : g_tbx;
    for (int i = 0; i < 8; i++) {
      alignas(64) bfloat16 v[64];
      for (int l = 0; l < 64; l++) v[l] = (bfloat16)(((int)(rnd() % 2001) - 1000) * (1.0f / 256));
      if (x[4]) for (int l = 0; l < 64; l++) v[l] = (bfloat16)0.0f;
      tb[i] = aie::load_v<64>(v);
    }
  }
  for (int i = 0; i < 1024 + 2 * GUARD; i++) ((uint16_t *)g_out)[i] = 0x7FC1;
  bfloat16 *out = g_out + GUARD;
  const uint32_t t0 = gstamp();
  ggk((const bfp16ebs8 *)g_a, (const bfp16ebs8 *)g_w, g_tbx, out, g_tba);
  const uint32_t t1 = gstamp();
  const uint32_t r0 = gstamp();
  gate_gemm((const bfp16ebs8 *)g_a, (const bfp16ebs8 *)g_w, g_tbx, g_ref);
  gate_gemm((const bfp16ebs8 *)g_a, (const bfp16ebs8 *)(g_w + 4608), g_tba, g_ref + 512);
  const uint32_t r1 = gstamp();
  int bad = 0, first = -1, guard = 0;
  for (int i = 0; i < 32; i++) y[200 + i] = 0;
  for (int i = 0; i < 1024; i++) {
    const uint16_t a = ((uint16_t *)out)[i], b = ((uint16_t *)g_ref)[i];
    if (a != b) { if (first < 0) { first = i; y[4] = a; y[5] = b; } bad++; y[200 + i / 32]++; }
  }
  for (int i = 0; i < GUARD; i++) {
    guard += ((uint16_t *)g_out)[i] != 0x7FC1;
    guard += ((uint16_t *)(out + 1024))[i] != 0x7FC1;
  }
  for (int i = 0; i < 64; i++) { y[16 + i] = ((int32_t *)out)[i]; y[80 + i] = ((int32_t *)g_ref)[i]; }
  y[0] = (int32_t)(t1 - t0); y[1] = bad; y[2] = (int32_t)(r1 - r0); y[3] = first; y[6] = guard;
}
