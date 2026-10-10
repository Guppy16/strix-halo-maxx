// round 15: one-core bench of r15/gen_ssx.py's ssx (log_a + exp + 1 - a^2 in one asm loop) against ssm9::loga_exp_a2 (deployed)
#include <aie_api/aie.hpp>
#include <stdint.h>
#define SEC_SSIG_AB __attribute__((section(".aie.bank2")))
#define SEC_SSIG_CD __attribute__((section(".aie.bank3")))
#define SEC_SSQRT_AB __attribute__((section(".aie.bank2")))
#define SEC_SSQRT_CD __attribute__((section(".aie.bank3")))
#include "r9/ssm_lut.h"
#include "r9/ssm9.h"
using B32 = aie::vector<bfloat16, 32>;
extern "C" void ssx(bfloat16 *g, const B32 *tsp, bfloat16 *om, const bfloat16 *tab);
#include "build/r15/ssx.h"
#define SSX_R4(b) b, b, b, b
#define SSX_ROW(b) SSX_R4(b), SSX_R4(b), SSX_R4(b), SSX_R4(b), SSX_R4(b), SSX_R4(b), SSX_R4(b), SSX_R4(b)
__attribute__((section(".aie.bank2"))) alignas(64) static uint16_t ssx_tab[SSX_NROWS * 32] = {SSX_ROWS};
#define SEC(b) __attribute__((section(".aie.bank" #b)))
constexpr int NV = 16, GUARD = 8;  // 16 vectors between guard zones of 8 vectors
SEC(1) alignas(64) static bfloat16 g_g[(NV + 2 * GUARD) * 32];
SEC(1) alignas(64) static bfloat16 g_om[(NV + 2 * GUARD) * 32];
SEC(2) alignas(64) static B32 g_tsp[8 + 8];
alignas(64) static bfloat16 g_rg[NV * 32];
alignas(64) static bfloat16 g_s[4][NV * 32];
static __attribute__((noinline)) uint32_t gstamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}
static uint32_t g_seed;
static float rnd01() {
  g_seed = g_seed * 1664525u + 1013904223u;
  return (g_seed >> 8) * (1.0f / (1 << 24));
}
static void fill(float spmax) {
  for (int i = 0; i < NV * 32; i++) g_rg[i] = (bfloat16)rnd01();  // gate_a = a sigmoid output in [0, 1)
  for (int cb = 0; cb < 8; cb++) {
    alignas(64) bfloat16 v[32];
    for (int l = 0; l < 32; l++) v[l] = (bfloat16)(rnd01() * spmax);  // softplus(a_param) >= 0
    g_tsp[cb] = aie::load_v<32>(v);
  }
}
// x: [1] seed, [2] sp max (quarters); y: [0] ssx cycles, [1] bad a, [2] bad om, [3] C++ cycles, [4] first bad index,
//    [5] got, [6] want, [7] guard elements changed (both buffers)
extern "C" void apb(int32_t *x, int32_t *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float spmax = x[2] * 0.25f;
  bfloat16 *g = g_g + GUARD * 32, *om = g_om + GUARD * 32;
  uint32_t t0 = 0, t1 = 0;
  for (int rep = 0; rep < 2; rep++) {
    g_seed = 7u + x[1];
    fill(spmax);
    for (int i = 0; i < (NV + 2 * GUARD) * 32; i++) ((uint16_t *)g_g)[i] = ((uint16_t *)g_om)[i] = 0x7FC1;
    for (int i = 0; i < NV * 32; i++) g[i] = g_rg[i];
    t0 = gstamp();
    ssx(g, g_tsp, om, (const bfloat16 *)((const uint8_t *)ssx_tab + SSX_TAB_OFF));
    t1 = gstamp();
  }
  const uint32_t r0 = gstamp();
  ssm9::loga_exp_a2(g_rg, g_tsp, g_s[0], g_s[1], g_s[2], g_s[3]);
  const uint32_t r1 = gstamp();
  int bad_a = 0, bad_o = 0, first = -1, guard = 0;
  for (int i = 0; i < NV * 32; i++) {
    const uint16_t a = ((uint16_t *)g)[i], b = ((uint16_t *)g_rg)[i];
    const uint16_t c = ((uint16_t *)om)[i], d = ((uint16_t *)g_s[1])[i];
    if (a != b) { if (first < 0) { first = i; y[5] = a; y[6] = b; } bad_a++; }
    if (c != d) { if (first < 0) { first = 100000 + i; y[5] = c; y[6] = d; } bad_o++; }
  }
  for (int i = 0; i < GUARD * 32; i++) {
    guard += ((uint16_t *)g_g)[i] != 0x7FC1;
    guard += ((uint16_t *)g_om)[i] != 0x7FC1;
    guard += ((uint16_t *)(g + NV * 32))[i] != 0x7FC1;
    guard += ((uint16_t *)(om + NV * 32))[i] != 0x7FC1;
  }
  for (int i = 0; i < 256; i++) y[16 + i] = ((int32_t *)g)[i];
  for (int i = 0; i < 256; i++) y[272 + i] = ((int32_t *)g_rg)[i];
  y[0] = (int32_t)(t1 - t0); y[1] = bad_a; y[2] = bad_o; y[3] = (int32_t)(r1 - r0); y[4] = first; y[7] = guard;
}
