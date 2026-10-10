// round 12: one-core bench of r12/gen_gup.py's gup against today's C++ gated path (mm8.cc acc_epi flags 2 / 3): bias_pcs
// on the up pieces, then per row block bf16(v x u) pushed as 4 bfp blocks. Built by r11/apb.py build --cc r12/gupb.cc.
#include "../mm8.cc"
#include "build/r12/gup_bench.h"

#define SEC(b) __attribute__((section(".aie.bank" #b)))
constexpr int NRB = 6;
SEC(1) alignas(64) static bfloat16 g_v[NRB * 512];  // gate: row block rb's 256 bf16 at the start of a 1 KB piece
SEC(2) alignas(64) static float g_u[NRB * 256];     // up: fp32 pieces, 1 KB per row block
alignas(64) static float g_u0[NRB * 256];
SEC(3) alignas(64) static bfloat16 g_bias[512];
SEC(3) alignas(64) static uint8_t g_out[NRB * 576 + 64];
alignas(64) static uint8_t g_ref[NRB * 576 + 64];

// mm8.cc bias_pcs (defined there only with MM_PIECES), verbatim
static __attribute__((noinline)) void bias_pcs(const float *__restrict src, bfloat16 *__restrict dst, const bfloat16 *__restrict bias,
                                               int npc, int dstride) {
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

extern "C" void gup(const bfloat16 *v, const float *u, bfp16ebs8 *dst, int trips, const bfloat16 *bias, int jump);

static inline __attribute__((always_inline)) uint32_t gstamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}

static uint32_t g_seed;
static float rnd(float scale) {  // LCG in [-scale, scale)
  g_seed = g_seed * 1664525u + 1013904223u;
  return ((int)(g_seed >> 9) - (1 << 22)) * (scale / (1 << 22));
}

// x: [1] stride (320: the stash; 576: gated_b), [2] seed
// y: [0] gup cycles, [1] mismatching bytes, [2] first mismatch, [3] C++ cycles, [4] got, [5] want
extern "C" void apb(int32_t *x, int32_t *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int stride = x[1];
  g_seed = 12345u + x[2];
  for (int i = 0; i < NRB * 512; i++) g_v[i] = (bfloat16)rnd(2.0f);
  for (int i = 0; i < NRB * 256; i++) g_u0[i] = rnd(4.0f);
  for (int i = 0; i < 512; i++) g_bias[i] = (bfloat16)rnd(0.5f);
  for (int i = 0; i < NRB * 256; i++) g_u[i] = g_u0[i];
  uint32_t t0 = 0, t1 = 0;
  for (int rep = 0; rep < 2; rep++) {  // gup reads u without changing it
    for (int i = 0; i < NRB * 576 + 64; i++) g_out[i] = 0xA5;
    t0 = gstamp();
    gup(g_v, g_u, (bfp16ebs8 *)g_out, 4 * NRB / GUP_U, g_bias + 256, stride - 288);
    t1 = gstamp();
  }
  for (int i = 0; i < NRB * 576 + 64; i++) g_ref[i] = 0xA5;
  const uint32_t r0 = gstamp();
  bias_pcs(g_u, (bfloat16 *)g_u, g_bias + 256, NRB, 512);  // today: the up half in place, then product + push per row block
  for (int rb = 0; rb < NRB; rb++) {
    const bfloat16 *v = g_v + 512 * rb, *u = (const bfloat16 *)(g_u + 256 * rb);
    B64 zv[4];
    for (int cb = 0; cb < 4; cb++)
      zv[cb] = aie::concat(BF(aie::mul(aie::load_v<32>(v + 64 * cb), aie::load_v<32>(u + 64 * cb))),
                           BF(aie::mul(aie::load_v<32>(v + 64 * cb + 32), aie::load_v<32>(u + 64 * cb + 32))));
    OutS ps((bfp16ebs8 *)(g_ref + stride * rb));
    for (int cb = 0; cb < 4; cb++) ps.push(A64(zv[cb]).template to_vector<bfp16ebs8>());
  }
  const uint32_t r1 = gstamp();
  int bad = 0, first = -1;
  for (int i = 0; i < NRB * 576 + 64; i++)
    if (g_out[i] != g_ref[i]) {
      if (first < 0) first = i;
      bad++;
    }
  y[0] = (int32_t)(t1 - t0);
  y[1] = bad;
  y[2] = first;
  y[3] = (int32_t)(r1 - r0);
  y[4] = first >= 0 ? g_out[first] : 0;
  y[5] = first >= 0 ? g_ref[first] : 0;
}
