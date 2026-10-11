// Clock probe (round 7): pure-register hardware loops with a runtime trip count. No loads or stores inside the loop,
// so no stall source exists (AIE cores stall only on memory / lock / stream access): cycles = trips x the loop's
// static bundle count (asmstat on the -S output). The host times dispatches of several trip counts; the slope is the
// clock. Two bodies of different bundle counts cross-check that the loop really runs at its static schedule.
#include <stdint.h>

#include <aie_api/aie.hpp>

using A32 = aie::accum<accfloat, 32>;
using B32 = aie::vector<bfloat16, 32>;

// scalar chain: an LCG step (mul + add) per trip; the multiply latency sets the bundle count
static __attribute__((noinline)) uint32_t chain_s(uint32_t x, int n) {
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; i++) x = x * 1664525u + 1013904223u;
  return x;
}

// vector chain: one dependent bf16 mac per trip (accumulator latency sets the bundle count)
static __attribute__((noinline)) A32 chain_v(A32 acc, B32 a, B32 b, int n) {
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; i++) acc = aie::mac(acc, a, b);
  return acc;
}

#ifdef STAMP
// npu-research P15: the core timer through the processor bus, which the runtime sequence enables first
// (Core_Processor_Bus 0x32038 bit 0); each read fenced so no other tm access shares its bundle
static inline __attribute__((always_inline)) uint32_t stamp() {
  __builtin_aie2p_sched_barrier();
  uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}
#endif

extern "C" void clk(int32_t *in, int32_t *out) {
  const int n = in[0], mode = in[1];
#ifdef STAMP
  if (mode >= 2) {  // 2 / 3: modes 0 / 1 between two timer reads; out[1] = core cycles
    const uint32_t t0 = stamp();
    if (mode == 2) out[0] = (int32_t)chain_s((uint32_t)in[2], n);
    else {
      B32 a = aie::load_v<32>((bfloat16 *)(in + 16)), b = aie::load_v<32>((bfloat16 *)(in + 32));
      aie::store_v((float *)(out + 16), chain_v(aie::zeros<accfloat, 32>(), a, b, n).to_vector<float>());
    }
    const uint32_t t1 = stamp();
    out[1] = (int32_t)(t1 - t0);
    return;
  }
#endif
  if (mode == 0) {
    out[0] = (int32_t)chain_s((uint32_t)in[2], n);
  } else {
    B32 a = aie::load_v<32>((bfloat16 *)(in + 16)), b = aie::load_v<32>((bfloat16 *)(in + 32));
    A32 acc = chain_v(aie::zeros<accfloat, 32>(), a, b, n);
    aie::store_v((float *)(out + 16), acc.to_vector<float>());
  }
}
