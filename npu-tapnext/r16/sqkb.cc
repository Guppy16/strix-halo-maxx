// round 16: one-core bench of r16/gen_sqk.py's sqk (exact bf16 sqrt in one asm loop) against ssm9::sqrt_lut_n (deployed),
// over every non-negative bf16 pattern 0x0000-0x7FFF (64 calls of 16 vectors), guard zones on both sides
#include <aie_api/aie.hpp>
#include <stdint.h>
#define SEC_SSIG_AB __attribute__((section(".aie.bank2")))
#define SEC_SSIG_CD __attribute__((section(".aie.bank3")))
#define SEC_SSQRT_AB __attribute__((section(".aie.bank2")))
#define SEC_SSQRT_CD __attribute__((section(".aie.bank2")))
#include "r9/ssm_lut.h"
#include "r9/ssm9.h"
#include "build/r16/sqk.h"
extern "C" void sqk(bfloat16 *v, const int16_t *ab, const int16_t *cd);
#define SEC(b) __attribute__((section(".aie.bank" #b)))
SEC(0) alignas(64) static int16_t sqk_ab[512] = SQK_TAB;
SEC(3) alignas(64) static int16_t sqk_cd[512] = SQK_TAB;
constexpr int NV = 16, GUARD = 8;
SEC(1) alignas(64) static bfloat16 g_v[(NV + 2 * GUARD) * 32];
alignas(64) static bfloat16 g_ref[NV * 32];
static __attribute__((noinline)) uint32_t gstamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}
// x: [1] first base, [2] number of 512-pattern blocks; y: [0] sqk cycles (last call), [1] bad, [2] C++ cycles, [3] first bad
//    pattern, [4] got, [5] want, [6] guard elements changed
extern "C" void apb(int32_t *x, int32_t *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  bfloat16 *v = g_v + GUARD * 32;
  uint32_t t0 = 0, t1 = 0, r0 = 0, r1 = 0;
  int bad = 0, first = -1, guard = 0;
  for (int blk = 0; blk < x[2]; blk++) {
    const int base = x[1] + 512 * blk;
    for (int i = 0; i < (NV + 2 * GUARD) * 32; i++) ((uint16_t *)g_v)[i] = 0x7FC1;
    for (int i = 0; i < NV * 32; i++) ((uint16_t *)v)[i] = ((uint16_t *)g_ref)[i] = (uint16_t)(base + i);
    t0 = gstamp();
    sqk(v, sqk_ab, sqk_cd);
    t1 = gstamp();
    r0 = gstamp();
    ssm9::sqrt_lut_n(g_ref, NV);
    r1 = gstamp();
    for (int i = 0; i < NV * 32; i++) {
      const uint16_t a = ((uint16_t *)v)[i], b = ((uint16_t *)g_ref)[i];
      if (a != b) {
        if (first < 0) { first = base + i; y[4] = a; y[5] = b; }
        bad++;
      }
    }
    for (int i = 0; i < GUARD * 32; i++) {
      guard += ((uint16_t *)g_v)[i] != 0x7FC1;
      guard += ((uint16_t *)(v + NV * 32))[i] != 0x7FC1;
    }
  }
  y[0] = (int32_t)(t1 - t0); y[1] = bad; y[2] = (int32_t)(r1 - r0); y[3] = first; y[6] = guard;
}
