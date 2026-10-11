// round 15: one-core bench of r15/gen_zk.py's zk against ew_ssm's z panel loop (verbatim below)
#include <aie_api/aie.hpp>
#include <stdint.h>
using B64 = aie::vector<bfloat16, 64>;
using A64 = aie::accum<accfloat, 64>;
using OutS = aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::none>;
#define BF(e) ((e).template to_vector<bfloat16>())
extern "C" void zk(const bfloat16 *hb, const bfloat16 *y, bfp16ebs8 *dst);
__attribute__((section(".aie.bank1"))) alignas(64) static bfloat16 g_hb[512];
__attribute__((section(".aie.bank3"))) alignas(64) static bfloat16 g_y[512];
__attribute__((section(".aie.bank0"))) alignas(64) static uint8_t g_out[1024];
alignas(64) static uint8_t g_ref[1024];
static __attribute__((noinline)) uint32_t gstamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}
static uint32_t g_seed;
static float rnd(float s) {
  g_seed = g_seed * 1664525u + 1013904223u;
  return ((int)(g_seed >> 9) - (1 << 22)) * (s / (1 << 22));
}
static __attribute__((noinline)) void ref(uint8_t *dst) {
  OutS zs((bfp16ebs8 *)dst);
  for (int cb = 0; cb < 8; cb++) {
    B64 y = aie::load_v<64>(g_y + cb * 64);
    B64 z = BF(aie::mul(aie::load_v<64>(g_hb + cb * 64), y));
    zs.push(A64(z).template to_vector<bfp16ebs8>());
  }
}
extern "C" void apb(int32_t *x, int32_t *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  g_seed = 13u + x[1];
  for (int i = 0; i < 512; i++) { g_hb[i] = (bfloat16)rnd(5.0f); g_y[i] = (bfloat16)rnd(3.0f); }
  uint32_t t0 = 0, t1 = 0;
  for (int rep = 0; rep < 2; rep++) {
    for (int i = 0; i < 1024; i++) g_out[i] = 0xA5;
    t0 = gstamp();
    zk(g_hb, g_y, (bfp16ebs8 *)g_out);
    t1 = gstamp();
  }
  for (int i = 0; i < 1024; i++) g_ref[i] = 0xA5;
  const uint32_t r0 = gstamp();
  ref(g_ref);
  const uint32_t r1 = gstamp();
  int bad = 0, past = 0;
  for (int i = 0; i < 576; i++) bad += g_out[i] != g_ref[i];
  for (int i = 576; i < 1024; i++) past += g_out[i] != 0xA5;
  y[0] = (int32_t)(t1 - t0); y[1] = bad; y[2] = (int32_t)(r1 - r0); y[3] = past;
}
