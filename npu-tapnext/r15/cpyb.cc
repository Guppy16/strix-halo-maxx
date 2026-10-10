// round 15: one-core bench of r15/gen_cpy64.py's cpy64 (bank 3 -> bank 0 like ew8's input ring -> output ring)
#include <aie_api/aie.hpp>
#include <stdint.h>
#include "build/r15/cpy64.h"
extern "C" void cpy64(const void *src, void *dst, int trips);
constexpr int NV = 32, GUARD = 8;
__attribute__((section(".aie.bank3"))) alignas(64) static uint16_t g_src[(NV + GUARD) * 32];
__attribute__((section(".aie.bank0"))) alignas(64) static uint16_t g_dst[(NV + 2 * GUARD) * 32];
static __attribute__((noinline)) uint32_t gstamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}
// x: [1] vectors (16 or 32); y: [0] cycles, [1] bad, [2] guard changed
extern "C" void apb(int32_t *x, int32_t *y) {
  const int n = x[1];
  uint32_t t0 = 0, t1 = 0;
  for (int rep = 0; rep < 2; rep++) {
    for (int i = 0; i < (NV + GUARD) * 32; i++) g_src[i] = (uint16_t)(i * 7 + 3 + x[1]);
    for (int i = 0; i < (NV + 2 * GUARD) * 32; i++) g_dst[i] = 0x7FC1;
    t0 = gstamp();
    cpy64(g_src, g_dst + GUARD * 32, n / CPY64_U);
    t1 = gstamp();
  }
  int bad = 0, guard = 0;
  for (int i = 0; i < n * 32; i++) bad += g_dst[GUARD * 32 + i] != g_src[i];
  for (int i = 0; i < GUARD * 32; i++) guard += g_dst[i] != 0x7FC1;
  for (int i = (GUARD + n) * 32; i < (NV + 2 * GUARD) * 32; i++) guard += g_dst[i] != 0x7FC1;
  y[0] = (int32_t)(t1 - t0); y[1] = bad; y[2] = guard;
}
