// round 12: one-core bench of r12/gen_copy.py's cpy32: the gated_b stash copy (18 rows of 288 B, 320 -> 576 B strides)
#include <aie_api/aie.hpp>
#include <stdint.h>
#include "build/r12/cpy32_bench.h"
extern "C" void cpy32(const void *src, void *dst, int trips, int src_jump, int dst_jump, int row_m1);
#define SEC(b) __attribute__((section(".aie.bank" #b)))
SEC(1) alignas(64) static uint8_t g_src[18 * 320];
SEC(2) alignas(64) static uint8_t g_dst[18 * 576 + 64];
static inline __attribute__((always_inline)) uint32_t gstamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}
// y: [0] cycles, [1] wrong bytes, [2] first wrong, [3] got, [4] want; [16..] the first 1024 B of dst
extern "C" void apb(int32_t *x, int32_t *y) {
  for (int i = 0; i < 18 * 320; i++) g_src[i] = (uint8_t)(i * 7 + (i >> 8));
  uint32_t t0 = 0, t1 = 0;
  for (int rep = 0; rep < 2; rep++) {
    for (int i = 0; i < 18 * 576 + 64; i++) g_dst[i] = 0xA5;
    t0 = gstamp();
    cpy32(g_src, g_dst + 288, 18 * 9 / CPY32_U, 320 - 256, 576 - 256, 8);
    t1 = gstamp();
  }
  int bad = 0, first = -1;
  for (int i = 0; i < 18 * 576 + 64; i++) {
    const int r = i / 576, c = i % 576;
    const uint8_t want = (r < 18 && c >= 288) ? g_src[r * 320 + c - 288] : 0xA5;
    if (g_dst[i] != want) {
      if (first < 0) { first = i; y[3] = g_dst[i]; y[4] = want; }
      bad++;
    }
  }
  y[0] = (int32_t)(t1 - t0);
  y[1] = bad;
  y[2] = first;
  for (int i = 0; i < 256; i++) y[16 + i] = ((int32_t *)g_dst)[i];
}
