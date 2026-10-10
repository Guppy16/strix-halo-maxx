// round 10: GELU loop bench on one core (r10/gxb.py). Buffers pinned to banks; cycles from the core timer.
#include "../mm8.cc"

#define SEC(b) __attribute__((section(".aie.bank" #b)))
SEC(1) alignas(64) static bfloat16 g_b1[4608];
SEC(2) alignas(64) static bfloat16 g_b2[4608];

static inline __attribute__((always_inline)) uint32_t gstamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}

static __attribute__((noinline)) void g_old(bfloat16 *v, int n, const float *ab, const float *cd) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int i = 0; i < 32 * n; i += 64) {
    B32 a = gelu_lut_i(aie::load_v<32>(v + i), ab, cd), b = gelu_lut_i(aie::load_v<32>(v + i + 32), ab, cd);
    aie::store_v(v + i, a);
    aie::store_v(v + i + 32, b);
  }
}
static __attribute__((noinline)) void g_ii(const bfloat16 *__restrict in, bfloat16 *__restrict out, int n, const float *ab, const float *cd) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
#pragma clang loop pipeline_initiation_interval(16)
  for (int i = 0; i < n; i++) {
    aie::store_v(out, gelu_lut_i(aie::load_v<32>(in), ab, cd));
    in += 32;
    out += 32;
  }
}

extern "C" void gxb(int32_t *x, int32_t *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int mode = x[0], n = x[1];
  for (int i = 0; i < 4608; i++) g_b1[i] = (bfloat16)(((i * 37) % 2048 - 1024) / 200.0f);
  uint32_t t0 = 0, t1 = 0;
  bfloat16 *res = g_b1;
  for (int rep = 0; rep < 2; rep++) {
    for (int i = 0; i < 4608; i++) g_b1[i] = (bfloat16)(((i * 37) % 2048 - 1024) / 200.0f);
    t0 = gstamp();
    for (int c = 0; c < 144 / n; c++) {
      if (mode == 0) g_old(g_b1 + 32 * n * c, n, sgelu_ab, sgelu_cd);
      else if (mode == 1) g_ii(g_b1 + 32 * n * c, g_b1 + 32 * n * c, n, sgelu_ab, sgelu_cd);
      else if (mode == 2) { g_ii(g_b1 + 32 * n * c, g_b2 + 32 * n * c, n, sgelu_ab, sgelu_cd); res = g_b2; }
      else if (mode == 3) { g_ii(g_b1 + 32 * n * c, g_b1 + 32 * n * c + (c & 1 ? -16 : 16), n, sgelu_ab, sgelu_cd); }
    }
    t1 = gstamp();
  }
  uint32_t h = 2166136261u;
  const uint16_t *w = (const uint16_t *)res;
  for (int i = 0; i < 4608; i++) h = (h ^ w[i]) * 16777619u;
  y[0] = (int32_t)(t1 - t0);
  y[1] = (int32_t)h;
}
