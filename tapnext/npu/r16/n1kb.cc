// round 16: one-core bench of r16/gen_n1k.py's n1k_ln / n1k_rms (norm pass 1 + statistics per unit) against norm_rows_k's C++
// pass-1 loop + statistics loop (verbatim) over one row block of 12 units at advancing (wrapping) ring offsets
#include <aie_api/aie.hpp>
#include <stdint.h>
using B32 = aie::vector<bfloat16, 32>;
using A32 = aie::accum<accfloat, 32>;
#define BF(e) ((e).template to_vector<bfloat16>())
#define FV(e) ((e).template to_vector<float>())
constexpr int KB = 1024, IN_B = 768 * 20, QS = 12;
extern "C" {
void n1k_ln_begin(float *, int, int);
void n1k_ln_step(const bfloat16 *, const bfloat16 *, const bfloat16 *, bfloat16 *, int, int, int, int);
void n1k_ln_end(float *);
void n1k_rms_begin(float *, int, int);
void n1k_rms_step(const bfloat16 *, const bfloat16 *, const bfloat16 *, bfloat16 *, int, int, int, int);
void n1k_rms_end(float *);
}
alignas(64) static uint32_t g_ring[IN_B / 4 + 64];
alignas(64) static bfloat16 g_st[QS * 512 + 256];
alignas(64) static bfloat16 g_rst[QS * 512];
alignas(64) static float g_acc[128];
alignas(64) static float g_racc[128];
static inline const bfloat16 *ring_r(const uint32_t *ring, int base, int off) {
  int p = base + off;
  if (p >= IN_B) p -= IN_B;
  return (const bfloat16 *)((const char *)ring + p);
}
static __attribute__((noinline)) uint32_t gstamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}
static __attribute__((noinline)) void ref_rb(const uint32_t *in_ring, int ib0, int n_in, int ln) {  // verbatim loops
  const B32 ones = aie::broadcast<bfloat16, 32>(1.0f);
  const int unit = n_in * KB;
  const int o1 = n_in > 1 ? KB : 0, o2 = n_in > 2 ? 2 * KB : 0;
  const B32 w1 = aie::broadcast<int16_t, 32>((int16_t)(n_in > 1 ? 0x3F80 : 0)).template cast_to<bfloat16>();
  const B32 w2 = aie::broadcast<int16_t, 32>((int16_t)(n_in > 2 ? 0x3F80 : 0)).template cast_to<bfloat16>();
  int ibq = ib0;
  for (int q = 0; q < QS; q++) {
    const int ib = ibq;
    bfloat16 *st = g_rst + q * 512;
    for (int cb = 0; cb < 8; cb++) {
      const bfloat16 *pa = ring_r(in_ring, ib, cb * 128), *pb = ring_r(in_ring, ib, o1 + cb * 128), *pc = ring_r(in_ring, ib, o2 + cb * 128);
      A32 a0 = aie::mul(aie::load_v<32>(pa), ones), a1 = aie::mul(aie::load_v<32>(pa + 32), ones);
      a0 = aie::mac(a0, aie::load_v<32>(pb), w1);
      a1 = aie::mac(a1, aie::load_v<32>(pb + 32), w1);
      a0 = aie::mac(a0, aie::load_v<32>(pc), w2);
      a1 = aie::mac(a1, aie::load_v<32>(pc + 32), w2);
      aie::store_v(st + cb * 64, BF(a0));
      aie::store_v(st + cb * 64 + 32, BF(a1));
    }
    ibq += unit;
    if (ibq >= IN_B) ibq -= IN_B;
  }
  A32 s0 = aie::zeros<accfloat, 32>(), s1 = s0, q0 = s0, q1 = s0;
  const bfloat16 *st = g_rst;
  if (ln) {
    for (int k = 0; k < QS * 8; k++) {
      const B32 x0 = aie::load_v<32>(st), x1 = aie::load_v<32>(st + 32);
      st += 64;
      s0 = aie::mac(s0, x0, ones);
      s1 = aie::mac(s1, x1, ones);
      q0 = aie::mac(q0, x0, x0);
      q1 = aie::mac(q1, x1, x1);
    }
  } else {
    for (int k = 0; k < QS * 8; k++) {
      const B32 x0 = aie::load_v<32>(st), x1 = aie::load_v<32>(st + 32);
      st += 64;
      q0 = aie::mac(q0, BF(aie::mul(x0, x0)), ones);
      q1 = aie::mac(q1, BF(aie::mul(x1, x1)), ones);
    }
  }
  aie::store_v(g_racc, FV(s0)); aie::store_v(g_racc + 32, FV(s1)); aie::store_v(g_racc + 64, FV(q0)); aie::store_v(g_racc + 96, FV(q1));
}
static uint32_t g_seed;
static uint32_t rnd() { g_seed = g_seed * 1664525u + 1013904223u; return g_seed >> 8; }
// x: [1] seed, [2] n_in, [3] ring offset, [4] ln; y: [0] asm cycles (12 units incl. per-unit pointer setup), [1] bad stage,
//    [2] C++ cycles, [3] bad acc, [4] first bad, [5] got, [6] want, [7] guard
extern "C" void apb(int32_t *x, int32_t *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  g_seed = 3u + x[1];
  for (int i = 0; i < IN_B / 2; i++) ((bfloat16 *)g_ring)[i] = (bfloat16)(((int)(rnd() % 4001) - 2000) * (1.0f / 64));
  for (int i = 0; i < QS * 512 + 256; i++) ((uint16_t *)g_st)[i] = 0x7FC1;
  for (int i = 0; i < 128; i++) g_acc[i] = 0.0f;
  const int n_in = x[2], ln = x[4], unit = n_in * KB;
  const int o[3] = {0, n_in > 1 ? KB : 0, n_in > 2 ? 2 * KB : 0};
  const int w1 = n_in > 1 ? 0x3F80 : 0, w2 = n_in > 2 ? 0x3F80 : 0;
  const uint32_t t0 = gstamp();
  if (ln) n1k_ln_begin(g_acc, w1, w2); else n1k_rms_begin(g_acc, w1, w2);
  int ib = x[3];
  for (int q = 0; q < QS; q++) {
    const bfloat16 *p[3];
    int dc[3];
    for (int s = 0; s < 3; s++) {
      int qq = ib + o[s];
      if (qq >= IN_B) qq -= IN_B;
      p[s] = (const bfloat16 *)((const char *)g_ring + qq);
      const int n1 = (IN_B - qq) >> 7;
      dc[s] = n1 >= 8 ? 0 : 8 - n1;
    }
    if (ln)
      n1k_ln_step(p[0], p[1], p[2], g_st + q * 512, dc[0], dc[1], dc[2], 64 - IN_B);
    else
      n1k_rms_step(p[0], p[1], p[2], g_st + q * 512, dc[0], dc[1], dc[2], 64 - IN_B);
    ib += unit;
    if (ib >= IN_B) ib -= IN_B;
  }
  if (ln) n1k_ln_end(g_acc); else n1k_rms_end(g_acc);
  const uint32_t t1 = gstamp();
  const uint32_t r0 = gstamp();
  ref_rb(g_ring, x[3], n_in, ln);
  const uint32_t r1 = gstamp();
  int bad = 0, bada = 0, first = -1, guard = 0;
  for (int i = 0; i < QS * 512; i++) {
    const uint16_t a = ((uint16_t *)g_st)[i], b = ((uint16_t *)g_rst)[i];
    if (a != b) { if (first < 0) { first = i; y[5] = a; y[6] = b; } bad++; }
  }
  // acc layout: s (s0 | s1) at 0, q (q0 | q1) at 64 floats; RMS: s unused
  for (int i = ln ? 0 : 64; i < 128; i++) {
    const uint32_t a = ((uint32_t *)g_acc)[i], b = ((uint32_t *)g_racc)[i];
    if (a != b) { if (first < 0) { first = 100000 + i; y[5] = a; y[6] = b; } bada++; }
  }
  for (int i = 0; i < 256; i++) guard += ((uint16_t *)(g_st + QS * 512))[i] != 0x7FC1;
  y[0] = (int32_t)(t1 - t0); y[1] = bad; y[2] = (int32_t)(r1 - r0); y[3] = bada; y[4] = first; y[7] = guard;
}
