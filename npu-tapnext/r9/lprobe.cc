// One-core probe of the round-9 SSM primitives (r9/lprobe.py): every bf16 bit pattern through the deployed nk8 helpers
// and the table versions (ssm9.h), cycles from the core timer (bus enabled by the runtime sequence).
#include <stdint.h>

#include <aie_api/aie.hpp>

#include "nk8.h"
#include "ssm_lut.h"
#include "ssm9.h"

#ifndef CH
#define CH 2048
#endif
using B32 = aie::vector<bfloat16, 32>;

static inline __attribute__((always_inline)) uint32_t stamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}

alignas(64) static bfloat16 g_om[CH];
static uint32_t g_canary[8];  // .bss: must read 0 on the first call (is .bss zero-filled when .data precedes it?)
static uint32_t g_calls;

extern "C" {
// mode: 0 copy, 1 sigmoid nk8, 2 sigmoid lut, 3 a nk8, 4 om nk8, 5 a lut, 6 om lut, 7 sqrt nk8, 8 sqrt lut
void lprobe(bfloat16 *in, bfloat16 *out, int32_t mode) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const nk8::ExpC kc;
  for (int i = 0; i < CH; i += 32) aie::store_v(out + i, aie::load_v<32>(in + i));
  const uint32_t t0 = stamp();
  if (mode == 1) nk8::sigmoid_n(out, CH / 32, kc);
  else if (mode == 2) ssm9::sigmoid_lut_n(out, CH / 32);
  else if (mode == 3 || mode == 4) nk8::exp_a2_n(out, g_om, CH / 32, kc);
  else if (mode == 5 || mode == 6) ssm9::a_om_lut_n(out, g_om, CH / 32, kc);
  else if (mode == 7) nk8::sqrt_n(out, CH / 32, kc);
  else if (mode == 8) ssm9::sqrt_lut_n(out, CH / 32);
  const uint32_t t1 = stamp();
  if (mode == 4 || mode == 6)
    for (int i = 0; i < CH; i += 32) aie::store_v(out + i, aie::load_v<32>(g_om + i));
  uint32_t *st = (uint32_t *)(out + CH);
  st[0] = t1 - t0;
  st[1] = (uint32_t)mode;
  if (g_calls == 0)
    for (int i = 0; i < 8; i++) st[2 + i] = g_canary[i];
  else
    for (int i = 0; i < 8; i++) st[2 + i] = 0xC0FFEE00u + i;
  st[10] = g_calls++;
}
}
