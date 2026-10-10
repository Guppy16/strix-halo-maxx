// round 12: one-core bench of r12/gen_pan.py's pan against mm8.cc's C++ panel push loop (flags 4 / 5 / 6)
#include "../mm8.cc"
#include "build/r12/pan_bench.h"
#define SEC(b) __attribute__((section(".aie.bank" #b)))
constexpr int NRB = 6;
SEC(1) alignas(64) static bfloat16 g_v[NRB * 512];  // pair 0: row block rb's 256 bf16 at the start of a 1 KB piece
SEC(2) alignas(64) static bfloat16 g_u[NRB * 512];  // pair 1
SEC(3) alignas(64) static uint8_t g_out[NRB * 576 + 2 * 1152 + 64];
alignas(64) static uint8_t g_ref[NRB * 576 + 2 * 1152 + 64];
constexpr int NB = NRB * 576 + 2 * 1152 + 64;
extern "C" void pan(const bfloat16 *v, const bfloat16 *u, bfp16ebs8 *dst, int trips, int jump, int group_m1);
static inline __attribute__((always_inline)) uint32_t gstamp() {
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
// x: [1] jump (bytes after a group), [2] rows per group, [3] seed; y: [0] pan cycles, [1] bad bytes, [2] first, [3] C++ cycles
extern "C" void apb(int32_t *x, int32_t *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int jump = x[1], group = x[2];
  g_seed = 777u + x[3];
  for (int i = 0; i < NRB * 512; i++) { g_v[i] = (bfloat16)rnd(3.0f); g_u[i] = (bfloat16)rnd(0.01f); }
  uint32_t t0 = 0, t1 = 0;
  for (int rep = 0; rep < 2; rep++) {
    for (int i = 0; i < NB; i++) g_out[i] = 0xA5;
    t0 = gstamp();
    pan(g_v, g_u, (bfp16ebs8 *)g_out, NRB / PAN_U, jump, group - 1);
    t1 = gstamp();
  }
  for (int i = 0; i < NB; i++) g_ref[i] = 0xA5;
  const uint32_t r0 = gstamp();
  int obj = 0, orb = 0;
  for (int rb = 0; rb < NRB; rb++) {
    const bfloat16 *v = g_v + 512 * rb, *u = g_u + 512 * rb;
    bfloat16 *dst = (bfloat16 *)(g_ref + obj * (group * 576 + jump) + orb * 576);
    if (++orb == group) { orb = 0; obj++; }
    OutS ps((bfp16ebs8 *)dst);
    for (int cb = 0; cb < 4; cb++) ps.push(A64(aie::load_v<64>(v + 64 * cb)).template to_vector<bfp16ebs8>());
    for (int cb = 0; cb < 4; cb++) ps.push(A64(aie::load_v<64>(u + 64 * cb)).template to_vector<bfp16ebs8>());
  }
  const uint32_t r1 = gstamp();
  int bad = 0, first = -1;
  for (int i = 0; i < NB; i++)
    if (g_out[i] != g_ref[i]) { if (first < 0) first = i; bad++; }
  y[0] = (int32_t)(t1 - t0); y[1] = bad; y[2] = first; y[3] = (int32_t)(r1 - r0);
}
