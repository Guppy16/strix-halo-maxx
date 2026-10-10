// round 12: one-core bench of r12/gen_sig.py's kernel-only sigk against r9/ssm9.h sigmoid_lut_n (the deployed SSM sigmoid)
#include <aie_api/aie.hpp>
#include <stdint.h>
#define SEC_SSIG_AB __attribute__((section(".aie.bank2")))
#define SEC_SSIG_CD __attribute__((section(".aie.bank3")))
#define SEC_SSQRT_AB __attribute__((section(".aie.bank2")))
#define SEC_SSQRT_CD __attribute__((section(".aie.bank3")))
#define SEC_SEXP_AB __attribute__((section(".aie.bank2")))
#define SEC_SEXP_CD __attribute__((section(".aie.bank3")))
#include "r9/ssm_lut.h"
#include "r9/ssm9.h"
#include "build/r12/sig_bench.h"
extern "C" void sigk(bfloat16 *v_last, int trips, const float *ab, const float *cd);
#define SEC(b) __attribute__((section(".aie.bank" #b)))
constexpr int NV = 32, GUARD = 16;  // 32 vectors (gate_x + gate_a) between guard zones of 16 vectors
SEC(1) alignas(64) static bfloat16 g_buf[(NV + 2 * GUARD) * 32];
alignas(64) static bfloat16 g_ref[NV * 32];
static __attribute__((noinline)) uint32_t gstamp() {
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
// x: [1] seed, [2] scale (quarters); y: [0] sigk cycles, [1] bad, [2] first, [3] C++ cycles, [4] got, [5] want,
//    [6] guard bytes changed before, [7] guard vectors changed after (expected: SIGK_GARBAGE)
extern "C" void apb(int32_t *x, int32_t *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  g_seed = 5u + x[1];
  const float scale = x[2] * 0.25f;
  bfloat16 *v = g_buf + GUARD * 32;
  uint32_t t0 = 0, t1 = 0;
  for (int rep = 0; rep < 2; rep++) {
    g_seed = 5u + x[1];
    for (int i = 0; i < (NV + 2 * GUARD) * 32; i++) ((uint16_t *)g_buf)[i] = 0x7FC1;
    for (int i = 0; i < NV * 32; i++) v[i] = (bfloat16)rnd(scale);
    t0 = gstamp();
    sigk(v + (NV - 1) * 32, NV / SIGK_U, ssig_ab, ssig_cd);
    t1 = gstamp();
  }
  g_seed = 5u + x[1];
  for (int i = 0; i < NV * 32; i++) g_ref[i] = (bfloat16)rnd(scale);
  const uint32_t r0 = gstamp();
  ssm9::sigmoid_lut_n(g_ref, NV);
  const uint32_t r1 = gstamp();
  int bad = 0, first = -1, before = 0, after = 0;
  for (int i = 0; i < NV * 32; i++) {
    const uint16_t a = ((uint16_t *)v)[i], b = ((uint16_t *)g_ref)[i];
    if (a != b) { if (first < 0) { first = i; y[4] = a; y[5] = b; } bad++; }
  }
  for (int i = 0; i < GUARD * 32; i++) before += ((uint16_t *)g_buf)[i] != 0x7FC1;
  for (int k = 0; k < GUARD; k++) {
    int ch = 0;
    for (int i = 0; i < 32; i++) ch |= ((uint16_t *)(v + NV * 32 + 32 * k))[i] != 0x7FC1;
    after += ch;
  }
  for (int i = 0; i < 512; i++) y[16 + i] = ((int32_t *)v)[i];
  for (int i = 0; i < 512; i++) y[528 + i] = ((int32_t *)g_ref)[i];
  y[0] = (int32_t)(t1 - t0); y[1] = bad; y[2] = first; y[3] = (int32_t)(r1 - r0); y[6] = before; y[7] = after;
}
