// round 11: one-core bench for hand-asm elementwise kernels (r11/apb.py). Probe k = extern "C" probe<k>(in, out, n, ab, cd)
// from the assembled .s; the driver runs it over 144 vectors of 32 bf16 in calls of n, times the calls with the core timer
// and compares every output element with the deployed C++ GELU (mm8.cc gelu_lut_n) on the same input.
#include "../mm8.cc"

#define GELU_FLAT_AB_SEC __attribute__((section(".aie.bank2")))  // the data area has no room for 2.3 KB more
namespace flat {  // round 12: flat-edge tables (r12/lutflat.py) for probes run with x[6] = 1; the reference keeps the deployed ones
#include "../gelu_lut_flat.h"
}
#define SEC(b) __attribute__((section(".aie.bank" #b)))
SEC(1) alignas(64) static bfloat16 g_in[4608];
SEC(2) alignas(64) static bfloat16 g_out[4608];
alignas(64) static bfloat16 g_ref[4608];

typedef void (*probe_t)(const bfloat16 *, bfloat16 *, int, const float *, const float *, int);
extern "C" {
#define P(k) void probe##k(const bfloat16 *, bfloat16 *, int, const float *, const float *, int);
P(0) P(1) P(2) P(3) P(4) P(5) P(6) P(7)
#undef P
}
static probe_t g_probes[8] = {probe0, probe1, probe2, probe3, probe4, probe5, probe6, probe7};

static inline __attribute__((always_inline)) uint32_t gstamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}

static int g_fill_mode, g_fill_base;
static void fill(bfloat16 *d, int seed) {
  if (g_fill_mode == 1) {  // round 12: bf16 bit patterns g_fill_base .. + 4607 (NaN / inf -> 0): 15 runs cover every finite bf16
    for (int i = 0; i < 4608; i++) {
      const uint16_t b = (uint16_t)(g_fill_base + i);
      ((uint16_t *)d)[i] = ((b & 0x7F80) == 0x7F80) ? 0 : b;
    }
    return;
  }
  for (int i = 0; i < 4608; i++) d[i] = (bfloat16)((((i + seed) * 37) % 2048 - 1024) / 200.0f);
}

// x: [0] probe, [1] n (vectors per call), [2] in place (out = in), [3] erf table, [4] layout (0: 144 contiguous vectors
// in calls of n; 1: the first 8 vectors of each of 9 pieces of 1 KB in one call, jump 576), [5] U (the probe's n argument
// is n / U loop trips; 0: n itself), [6] flat-edge tables for the probe, [7] fill mode (1: bit patterns from [8])
// y: [0] cycles for all vectors (second repetition), [1] mismatching elements, [2] first mismatch index, [3] got, [4] want
extern "C" void apb(int32_t *x, int32_t *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int k = x[0], inplace = x[2], pieces = x[4], U = x[5];
  const int n = pieces ? 72 : x[1];
  const float *ab = x[3] ? serf_ab : sgelu_ab, *cd = x[3] ? serf_cd : sgelu_cd;
  const float *pab = x[6] ? (x[3] ? flat::serf_ab : flat::sgelu_ab) : ab, *pcd = x[6] ? (x[3] ? flat::serf_cd : flat::sgelu_cd) : cd;
  g_fill_mode = x[7];
  g_fill_base = x[8];
  fill(g_ref, 0);
  if (pieces)
    for (int c = 0; c < 9; c++) gelu_lut_n(g_ref + 512 * c, 8, ab, cd);
  else
    for (int c = 0; c < 144 / n; c++) gelu_lut_n(g_ref + 32 * n * c, n, ab, cd);
  bfloat16 *out = inplace ? g_in : g_out;
  const int narg = U ? n / U : n;
  uint32_t t0 = 0, t1 = 0;
  for (int rep = 0; rep < 2; rep++) {
    fill(g_in, 0);
    for (int i = 0; i < 4608; i++) ((uint16_t *)g_out)[i] = 0x7FC1;  // sentinel (a NaN)
    t0 = gstamp();
    if (pieces)
      g_probes[k](g_in, out, narg, pab, pcd, 576);
    else
      for (int c = 0; c < 144 / n; c++) g_probes[k](g_in + 32 * n * c, out + 32 * n * c, narg, pab, pcd, 64);
    t1 = gstamp();
  }
  int bad = 0, first = -1;
  const uint16_t *a = (const uint16_t *)out, *b = (const uint16_t *)g_ref;
  for (int i = 0; i < 4608; i++)
    if ((!pieces || (i & 511) < 256) && a[i] != b[i]) {  // pieces: only the processed first half of each 1 KB piece
      if (first < 0) first = i;
      bad++;
    }
  y[0] = (int32_t)(t1 - t0);
  y[1] = bad;
  y[2] = first;
  y[3] = first >= 0 ? a[first] : 0;
  y[4] = first >= 0 ? b[first] : 0;
  for (int i = 0; i < 1024; i++) y[16 + i] = (int32_t)(a[2 * i] | ((uint32_t)a[2 * i + 1] << 16));  // the first 2048 outputs
}
