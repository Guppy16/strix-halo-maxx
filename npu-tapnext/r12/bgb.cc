// round 12: one-core bench of r12/gen_bgelu.py's bgelu (bias + GELU fused, flat-edge tables) against today's path:
// bias_pcs (bf16(acc + bias) at each piece's start) + gelu_lut_n (the deployed tables), on NP pieces of 1 KB
#include "../mm8.cc"
#include "build/r12/bgelu_bench.h"
#define GELU_FLAT_AB_SEC __attribute__((section(".aie.bank2")))
namespace flat {
#include "../gelu_lut_flat.h"
}
#define SEC(b) __attribute__((section(".aie.bank" #b)))
constexpr int NP = 4;
SEC(1) alignas(64) static float g_acc[NP * 256];
SEC(2) alignas(64) static float g_ref[NP * 256];
SEC(3) alignas(64) static float g_src[NP * 256];
SEC(3) alignas(64) static bfloat16 g_bias[256];
extern "C" void bgelu(const float *acc, bfloat16 *out, int trips, const float *ab, const float *cd, const bfloat16 *bias);
static __attribute__((noinline)) void bias_pcs(const float *__restrict src, bfloat16 *__restrict dst, const bfloat16 *__restrict bias,
                                               int npc, int dstride) {  // mm8.cc (MM_PIECES), verbatim
  const B32 ones = aie::broadcast<bfloat16, 32>(1.0f);
  AIE_LOOP_MIN_ITERATION_COUNT(3)
  for (int i = 0; i < npc; i += 2) {
#pragma unroll
    for (int v = 0; v < 8; v++) {
      aie::store_v(dst + 32 * v, BF(aie::mac(A32(aie::load_v<32>(src + 32 * v)), aie::load_v<32>(bias + 32 * v), ones)));
      aie::store_v(dst + dstride + 32 * v, BF(aie::mac(A32(aie::load_v<32>(src + 256 + 32 * v)), aie::load_v<32>(bias + 32 * v), ones)));
    }
    src += 512;
    dst += 2 * dstride;
  }
}
static __attribute__((noinline)) uint32_t gstamp() {  // noinline: an inlined sched_barrier beside a pipelined loop crashes Peano
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}
static uint32_t g_seed;
static float rnd(float scale) {
  g_seed = g_seed * 1664525u + 1013904223u;
  return ((int)(g_seed >> 9) - (1 << 22)) * (scale / (1 << 22));
}
// x: [1] erf table, [2] seed, [3] scale; y: [0] bgelu cycles, [1] bad bf16, [2] first, [3] C++ cycles, [4] got, [5] want
extern "C" void apb(int32_t *x, int32_t *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bool erf = x[1];
  g_seed = 99u + x[2];
  const float scale = x[3] ? x[3] * 0.25f : 6.0f;
  for (int i = 0; i < NP * 256; i++) g_src[i] = rnd(scale);
  for (int i = 0; i < 256; i++) g_bias[i] = (bfloat16)rnd(1.0f);
  uint32_t t0 = 0, t1 = 0;
  for (int rep = 0; rep < 2; rep++) {
    for (int i = 0; i < NP * 256; i++) g_acc[i] = g_src[i];
    t0 = gstamp();
    bgelu(g_acc, (bfloat16 *)g_acc, 8 * NP / BGELU_U, erf ? flat::serf_ab : flat::sgelu_ab, erf ? flat::serf_cd : flat::sgelu_cd, g_bias);
    t1 = gstamp();
  }
  for (int i = 0; i < NP * 256; i++) g_ref[i] = g_src[i];
  const uint32_t r0 = gstamp();
  bias_pcs(g_ref, (bfloat16 *)g_ref, g_bias, NP, 512);
  for (int p = 0; p < NP; p++) gelu_lut_n((bfloat16 *)(g_ref + 256 * p), 8, erf ? serf_ab : sgelu_ab, erf ? serf_cd : sgelu_cd);
  const uint32_t r1 = gstamp();
  int bad = 0, first = -1;
  for (int p = 0; p < NP; p++)
    for (int i = 0; i < 256; i++) {  // the bf16 results at each piece's start
      const uint16_t a = ((uint16_t *)(g_acc + 256 * p))[i], b = ((uint16_t *)(g_ref + 256 * p))[i];
      if (a != b) { if (first < 0) { first = 256 * p + i; y[4] = a; y[5] = b; } bad++; }
    }
  y[0] = (int32_t)(t1 - t0); y[1] = bad; y[2] = first; y[3] = (int32_t)(r1 - r0);
}
