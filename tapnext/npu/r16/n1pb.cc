// round 16: one-core bench of r16/gen_n1p.py's n1p (norm pass 1 for one unit) against norm_rows_k's C++ pass-1 loop (verbatim,
// with ring_r), on a 15 KB ring with the unit placed at chosen ring offsets (wraps inside streams), n_in 1 / 2 / 3
#include <aie_api/aie.hpp>
#include <stdint.h>
using B32 = aie::vector<bfloat16, 32>;
using A32 = aie::accum<accfloat, 32>;
#define BF(e) ((e).template to_vector<bfloat16>())
constexpr int KB = 1024, IN_B = 768 * 20;
extern "C" void n1p(const bfloat16 *pa, const bfloat16 *pb, const bfloat16 *pc, bfloat16 *st, int dna, int dnb, int dnc, int w1, int w2, int m);
#define SEC(b) __attribute__((section(".aie.bank" #b)))
alignas(64) static uint32_t g_ring[IN_B / 4 + 64];
alignas(64) static bfloat16 g_st[512 + 2 * 256];
alignas(64) static bfloat16 g_ref[512];
static inline const bfloat16 *ring_r(const uint32_t *ring, int base, int off) {
  int p = base + off;
  if (p >= IN_B) p -= IN_B;
  return (const bfloat16 *)((const char *)ring + p);
}
static __attribute__((noinline)) void ref_pass1(const uint32_t *in_ring, int ib, int n_in, bfloat16 *st) {  // verbatim loop body
  const B32 ones = aie::broadcast<bfloat16, 32>(1.0f);
  const int o1 = n_in > 1 ? KB : 0, o2 = n_in > 2 ? 2 * KB : 0;
  const B32 w1 = aie::broadcast<int16_t, 32>((int16_t)(n_in > 1 ? 0x3F80 : 0)).template cast_to<bfloat16>();
  const B32 w2 = aie::broadcast<int16_t, 32>((int16_t)(n_in > 2 ? 0x3F80 : 0)).template cast_to<bfloat16>();
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
}
static uint32_t g_seed;
static uint32_t rnd() { g_seed = g_seed * 1664525u + 1013904223u; return g_seed >> 8; }
// the call as norm_rows_k would make it: per stream the pointer and (pieces before the ring end) - 1
static __attribute__((noinline)) uint32_t gstamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}
static uint32_t g_tc;  // cycles of the n1p call alone
static void call_n1p(const uint32_t *ring, int ib, int n_in, bfloat16 *st) {
  const int o[3] = {0, n_in > 1 ? KB : 0, n_in > 2 ? 2 * KB : 0};
  const bfloat16 *p[3];
  int dn[3];
  for (int s = 0; s < 3; s++) {
    int q = ib + o[s];
    if (q >= IN_B) q -= IN_B;
    p[s] = (const bfloat16 *)((const char *)ring + q);
    const int n1 = (IN_B - q) / 128;
    dn[s] = n1 >= 8 ? 0 : 8 - n1;  // the 2D counter's start (r16/gen_n1p.py)
  }
  const int w1 = n_in > 1 ? 0x3F80 : 0, w2 = n_in > 2 ? 0x3F80 : 0;
  const uint32_t a = gstamp();
  n1p(p[0], p[1], p[2], st, dn[0], dn[1], dn[2], w1, w2, 64 - IN_B);
  g_tc = gstamp() - a;
}
// x: [1] seed, [2] n_in, [3] ring offset of the unit (multiple of 128); y: [0] n1p cycles, [1] bad, [2] C++ cycles, [3] first, [4] got, [5] want, [6] guard
extern "C" void apb(int32_t *x, int32_t *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  g_seed = 3u + x[1];
  for (int i = 0; i < IN_B / 2; i++) ((bfloat16 *)g_ring)[i] = (bfloat16)(((int)(rnd() % 4001) - 2000) * (1.0f / 64));
  for (int i = 0; i < 512 + 2 * 256; i++) ((uint16_t *)g_st)[i] = 0x7FC1;
  bfloat16 *st = g_st + 256;
  const uint32_t t0 = gstamp();
  call_n1p(g_ring, x[3], x[2], st);
  const uint32_t t1 = gstamp();
  const uint32_t r0 = gstamp();
  ref_pass1(g_ring, x[3], x[2], g_ref);
  const uint32_t r1 = gstamp();
  int bad = 0, first = -1, guard = 0;
  for (int i = 0; i < 512; i++) {
    const uint16_t a = ((uint16_t *)st)[i], b = ((uint16_t *)g_ref)[i];
    if (a != b) { if (first < 0) { first = i; y[4] = a; y[5] = b; } bad++; }
  }
  for (int i = 0; i < 256; i++) { guard += ((uint16_t *)g_st)[i] != 0x7FC1; guard += ((uint16_t *)(st + 512))[i] != 0x7FC1; }
  y[7] = (int32_t)g_tc;
  y[0] = (int32_t)(t1 - t0); y[1] = bad; y[2] = (int32_t)(r1 - r0); y[3] = first; y[6] = guard;
}
