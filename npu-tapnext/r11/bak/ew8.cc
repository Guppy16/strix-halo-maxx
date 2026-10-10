// design8 elementwise kernels for the row-3 core of a column: norms (+ residual sums), the patch embedding and the
// SSM. One call = one whole op; the kernel drives its input / output rings with lock intrinsics (768 B slots).
//
// Input ring: a shim -> core packet flow fills SLOT_B bytes per BD execution (one BD iterating over the ring) and
// releases L_IN_FILL per slot; the kernel consumes BYTE units (in_need / in_done) and releases L_IN_FREE per whole
// slot. Output ring: the kernel acquires L_OUT_FREE per slot before writing it and releases L_OUT_FILL when full.
//
// Activations arrive as Q16 units: an 8-row x 64-col k-step piece is 1024 B of bf16 in [col-block 8][8 rows][8 cols]
// order, so a 32-lane vector holds 4 rows x 8 columns and an 8x8 block is one bfp16ebs8 panel.
#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"
#include "nk8.h"
#ifdef SSM_LUT  // round 9: the RG-LRU sigmoid and sqrt by table lookup (r9/ssm9.h, r9/ssm_lut.h; numerics r9/ssmemu.py)
// the free L1 left by IRON's placement (ldScripts: bank0 0x73C00 1 KB, bank2 0x7AC00 5 KB, bank3 0x7FC00 1 KB) and 2.5 KB
// of the arena: the sigmoid copies in banks 2 and 1 (one bank would cost +8 cycles per 32 lanes, r9/lprobe.py), the
// sqrt copies in banks 0 and 3
#define SEC_SSIG_AB  // the data area (.data precedes the arena: bank 1); g_arena gives up 2,560 B for it
#define SEC_SSIG_CD __attribute__((section(".aie.bank2")))
#define SEC_SSQRT_AB __attribute__((section(".aie.bank0")))
#define SEC_SSQRT_CD __attribute__((section(".aie.bank3")))
#include "r9/ssm_lut.h"
#include "r9/ssm9.h"
#endif

constexpr int SLOT_B = 768;
constexpr int IN_B = SLOT_B * EW_IN_SLOTS;
constexpr int OUT_B = SLOT_B * EW_OUT_SLOTS;
constexpr int D = 768;
constexpr int QS = D / 64;   // 12 k-steps per row
constexpr int KB = 1024;     // one k-step piece of 8 rows
constexpr int PB = 72;       // bytes per bfp panel
constexpr int PANB = 8 * PB; // 8 panels = one k-step's companion
constexpr int RBN = 18;      // row blocks per column
constexpr int UNIT_B = KB + PANB;  // one k-step's output: X' + panels

using B32 = aie::vector<bfloat16, 32>;
using B64 = aie::vector<bfloat16, 64>;
using B16 = aie::vector<bfloat16, 16>;
using B8 = aie::vector<bfloat16, 8>;
using F16 = aie::vector<float, 16>;
using F32 = aie::vector<float, 32>;
using I16 = aie::vector<int32_t, 16>;
using A16 = aie::accum<accfloat, 16>;
using A32 = aie::accum<accfloat, 32>;
using A64 = aie::accum<accfloat, 64>;
using OutS = aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::none>;
using MM = aie::mmul<8, 8, 8, bfp16ebs8>;
using BP = aie::block_vector<bfp16ebs8, 64>;
template <aie_dm_resource R>
using InS = aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, 64, R>;
#define BF(e) ((e).template to_vector<bfloat16>())
#define FV(e) ((e).template to_vector<float>())

// ---- ring protocol ---------------------------------------------------------------------------------------------
static int g_in_b = 0;        // byte position of the next unread input byte
static int g_in_avail = 0;    // bytes acquired but not consumed
static int g_in_consumed = 0; // bytes consumed since the last slot release (< SLOT_B after each release)
static int g_out_b = 0;       // byte position of the next output byte
static int g_out_slot_b = 0;  // bytes written into the current output slot

#ifdef PROF  // round 8 timing build: core-timer phase totals (the runtime sequence enables the processor bus)
static inline __attribute__((always_inline)) uint32_t stamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}
static uint32_t g_ph[12], g_tlast, g_nchunk;
// a call, not inline code: sched barriers inside the pipelined chunk loop crashed Peano (AIE Region assertion)
static __attribute__((noinline)) void ph(int i) {
  const uint32_t t = stamp();
  g_ph[i] += t - g_tlast;
  g_tlast = t;
}
#define PH(i) ph(i)
#else
#define PH(i) ((void)0)
#endif
static inline void lacq(const void *p, int id, int n) { acquire_greater_equal(p, id + 48, n); }
static inline void lrel(void *p, int id, int n) { release(p, id + 48, n); }

static __attribute__((noinline, minsize)) void in_need(const uint32_t *ring, int bytes) {  // make `bytes` of unread input resident
  int n = 0;
  while (g_in_avail < bytes) {  // (no division: __divsi3 costs program memory)
    g_in_avail += SLOT_B;
    n++;
  }
  if (n) lacq(ring, L_IN_FILL, n);
}

static __attribute__((noinline, minsize)) void in_done(uint32_t *ring, int bytes) {  // consume `bytes`, release whole slots
  g_in_avail -= bytes;
  g_in_b += bytes;
  if (g_in_b >= IN_B) g_in_b -= IN_B;
  g_in_consumed += bytes;
  int n = 0;
  while (g_in_consumed >= SLOT_B) {
    g_in_consumed -= SLOT_B;
    n++;
  }
  if (n) lrel(ring, L_IN_FREE, n);
}

// pointer to byte `off` of the unread input (wraps; 64 B vectors never straddle the end: IN_B % 64 == 0)
static inline const bfloat16 *in_at(const uint32_t *ring, int off) {
  int p = g_in_b + off;
  if (p >= IN_B) p -= IN_B;
  return (const bfloat16 *)((const char *)ring + p);
}

static __attribute__((noinline)) void out64(uint32_t *ring, B32 v) {  // append one 64 B vector to the output ring
  if (g_out_slot_b == 0) lacq(ring, L_OUT_FREE, 1);
  aie::store_v((bfloat16 *)((char *)ring + g_out_b), v);
  g_out_b += 64;
  g_out_slot_b += 64;
  if (g_out_slot_b == SLOT_B) {
    lrel(ring, L_OUT_FILL, 1);
    g_out_slot_b = 0;
  }
  if (g_out_b == OUT_B) g_out_b = 0;
}

static __attribute__((noinline)) void out_bytes(uint32_t *ring, const void *src, int bytes) {
  const bfloat16 *s = (const bfloat16 *)src;
  for (int i = 0; i < bytes / 64; i++) out64(ring, aie::load_v<32>(s + 32 * i));
}

static __attribute__((noinline, minsize)) void in_copy(const uint32_t *ring, void *dst, int off, int bytes) {  // ring -> L1 copy
  bfloat16 *d = (bfloat16 *)dst;
  for (int i = 0; i < bytes / 64; i++) aie::store_v(d + 32 * i, aie::load_v<32>(in_at(ring, off + 64 * i)));
}

// ---- helpers -----------------------------------------------------------------------------------------------------
static inline B32 rep4(const bfloat16 *v8) {  // 8 per-column values repeated for 4 rows
  B8 s = aie::load_v<8>(v8);
  return aie::concat(s, s, s, s);
}

static inline void split32(const A32 &a, B32 &hi, B32 &lo) {  // fp32 accumulator -> bf16 hi + lo
  hi = BF(a);
  lo = BF(aie::mac(a, hi, (bfloat16)-1.0f));
}

static inline void split3(const A32 &a, B32 &h, B32 &m, B32 &l) {  // fp32 accumulator -> three bf16 parts (exact)
  h = BF(a);
  A32 r = aie::mac(a, h, (bfloat16)-1.0f);
  m = BF(r);
  l = BF(aie::mac(r, m, (bfloat16)-1.0f));
}

// a * c for an fp32 accumulator and a constant given as three bf16 parts: ~2^-20 relative (the 2^-24 terms dropped)
static __attribute__((noinline, minsize)) A32 mul3(const A32 &a, bfloat16 ch, bfloat16 cm, bfloat16 cl) {
  B32 h, m, l;
  split3(a, h, m, l);
  A32 r = aie::mul(h, ch);
  r = aie::mac(r, h, cm);
  r = aie::mac(r, m, ch);
  r = aie::mac(r, h, cl);
  r = aie::mac(r, m, cm);
  return aie::mac(r, l, ch);
}

// every lane of a [4 rows][8] fp32 accumulator gets its row's sum: a butterfly over each row's 8 lanes (shuffles and
// accumulator adds, all native; the per-row reduce_add it replaces ran on emulated fp32)
static __attribute__((noinline, minsize)) A32 rowsum8(A32 a) {
  for (int st = 4; st >= 1; st /= 2) {
    F32 v = FV(a);
    auto [e, o] = aie::interleave_unzip(v, v, st);
    a = aie::add(a, A32(aie::interleave_zip(o, e, st).first));
  }
  return a;
}

// 1/sqrt(vh + vl) as an fp32 accumulator: int16 bit-trick seed (3.4%), one Newton step in bf16, two carried in
// fp32 as hi/lo pairs (y += y (1 - v y^2) / 2): ~1e-7 relative
static __attribute__((noinline, minsize)) A32 rsqrt_i(B32 vh, B32 vl) {
  const B32 ones = aie::broadcast<bfloat16, 32>(1.0f);
  const A32 c1(aie::broadcast<float, 32>(1.0f)), c15(aie::broadcast<float, 32>(1.5f));
  vh = aie::max(vh, aie::broadcast<bfloat16, 32>(1e-30f));
  B32 y = aie::sub(aie::broadcast<int16_t, 32>((int16_t)0x5F37), aie::downshift(vh.template cast_to<int16_t>(), 1)).template cast_to<bfloat16>();
  B32 hv = BF(aie::mul(vh, (bfloat16)0.5f));
  y = BF(aie::mul(y, BF(aie::msc(c15, hv, BF(aie::mul(y, y))))));
  B32 yl = aie::zeros<bfloat16, 32>();
  for (int i = 0; i < 2; i++) {
    A32 q = aie::mul(y, y);
    q = aie::mac(q, y, BF(aie::mul(yl, (bfloat16)2.0f)));
    B32 qh, ql;
    split32(q, qh, ql);
    A32 r = aie::msc(c1, vh, qh);
    r = aie::msc(r, vh, ql);
    r = aie::msc(r, vl, qh);
    A32 yn = aie::mul(y, ones);
    yn = aie::mac(yn, yl, ones);
    yn = aie::mac(yn, y, BF(aie::mul(BF(r), (bfloat16)0.5f)));
    split32(yn, y, yl);
  }
  return aie::mac(A32(aie::mul(y, ones)), yl, ones);
}

// 1/768 and 1/768^2 as three bf16 parts each (exact sums)
constexpr float INV_D[3] = {0.00130462646484375f, -2.5480985641479492e-06f, 4.976755008101463e-09f};
constexpr float INV_D2[3] = {1.6987323760986328e-06f, -3.3178366720676422e-09f, 6.480149750132114e-12f};

// RMSNorm statistics of 4 rows (torch's bf16 numerics): q = sum of bf16(x^2) per row (every lane) -> bf16 rstd
static __attribute__((noinline, minsize)) B32 rms_rstd(A32 q, B32 eps) {
  const B32 ones = aie::broadcast<bfloat16, 32>(1.0f);
  B32 mean = BF(mul3(q, INV_D[0], INV_D[1], INV_D[2]));
  B32 ve = BF(aie::mac(A32(aie::mul(mean, ones)), eps, ones));
  return BF(rsqrt_i(ve, aie::zeros<bfloat16, 32>()));
}

// LayerNorm statistics of 4 rows from exact fp32 row sums s, q (every lane): rstd (hi/lo) and nb = -mean rstd.
// var = (768 q - s^2) / 768^2 with every product exact and fp32 accumulation, as the fp32 code it replaces
static __attribute__((noinline, minsize)) void ln_stats(A32 s, A32 q, B32 eps, B32 &rh, B32 &rl, A32 &nb) {
  const B32 ones = aie::broadcast<bfloat16, 32>(1.0f);
  B32 sh, sm, sl, qh, qm, ql;
  split3(s, sh, sm, sl);
  split3(q, qh, qm, ql);
  A32 d = aie::mul(qh, (bfloat16)768.0f);
  d = aie::mac(d, qm, (bfloat16)768.0f);
  d = aie::mac(d, ql, (bfloat16)768.0f);
  d = aie::msc(d, sh, sh);
  d = aie::msc(d, sh, BF(aie::mul(sm, (bfloat16)2.0f)));
  d = aie::msc(d, sh, BF(aie::mul(sl, (bfloat16)2.0f)));
  d = aie::msc(d, sm, sm);
  A32 var = aie::mac(mul3(d, INV_D2[0], INV_D2[1], INV_D2[2]), eps, ones);
  B32 vh, vl;
  split32(var, vh, vl);
  A32 r = rsqrt_i(vh, vl);
  split32(r, rh, rl);
  // nb = -mean rstd from three-part factors (~2^-22): rows whose mean is large against their spread need it
  B32 mh, mm, ml, r0, r1, r2;
  split3(mul3(s, INV_D[0], INV_D[1], INV_D[2]), mh, mm, ml);
  split3(r, r0, r1, r2);
  nb = aie::negmul(mh, r0);
  nb = aie::msc(nb, mh, r1);
  nb = aie::msc(nb, mm, r0);
  nb = aie::msc(nb, mh, r2);
  nb = aie::msc(nb, mm, r1);
  nb = aie::msc(nb, ml, r0);
}

// one static arena shared by the kernels (never two ops live at once): norm = stage 12,288 + params 3072;
// patch = 3 weight tiles 13,824 + 24 A panels 1728 + bias table 1024; ssm = see ew_ssm
#ifdef SSM_LUT  // round 9: the SSM layout below needs 21,568 B; the sigmoid table's second copy takes the rest (bank 1)
alignas(64) static uint8_t g_arena[24576 - 2560];
#else
alignas(64) static uint8_t g_arena[24576];
#endif
static bfloat16 *const g_stage = (bfloat16 *)g_arena;          // one chunk's X' (8 rows x 768) in k-step pieces
static bfloat16 *const g_p0 = (bfloat16 *)(g_arena + 12288);   // norm params: (scale + 1) | gamma
static bfloat16 *const g_p1 = (bfloat16 *)(g_arena + 13824);   // beta
alignas(64) static bfp16ebs8 g_panels[KB];  // PANB panels + a zero pad (the SSM copies 1 KB)

// output ring: reserve `bytes` from the current position (acquire the slots not yet held), then commit them
static __attribute__((noinline, minsize)) void out_begin(uint32_t *ring, int bytes) {
  int have = g_out_slot_b ? SLOT_B - g_out_slot_b : 0, n = 0;
  while (have < bytes) {
    have += SLOT_B;
    n++;
  }
  if (n) lacq(ring, L_OUT_FREE, n);
}
static __attribute__((noinline, minsize)) void out_end(uint32_t *ring, int bytes) {
  g_out_b += bytes;
  if (g_out_b >= OUT_B) g_out_b -= OUT_B;
  int t = g_out_slot_b + bytes, n = 0;
  while (t >= SLOT_B) {
    t -= SLOT_B;
    n++;
  }
  g_out_slot_b = t;
  if (n) lrel(ring, L_OUT_FILL, n);
}
static inline bfloat16 *ring_w(uint32_t *ring, int base, int off) {  // 64 B vectors never straddle the ring end
  int p = base + off;
  if (p >= OUT_B) p -= OUT_B;
  return (bfloat16 *)((char *)ring + p);
}
static inline const bfloat16 *ring_r(const uint32_t *ring, int base, int off) {  // 128 B pieces never straddle
  int p = base + off;                                                            // (every op moves 128 B multiples)
  if (p >= IN_B) p -= IN_B;
  return (const bfloat16 *)((const char *)ring + p);
}

// all 18 row blocks of a norm op. Per row block: pass 1 sums the n_in streams into X' (stage) and the row sums,
// the statistics run on 32 lanes (every lane holds its row's value), pass 2 writes per k-step [X' | panels] straight
// into the output ring (to_x: the LN result replaces X', the encoder norm)
template <bool LN>
static __attribute__((noinline)) void norm_rows(uint32_t *in_ring, uint32_t *out_ring, int n_in, int to_x, B32 eps) {
  const B32 ones = aie::broadcast<bfloat16, 32>(1.0f);
  const int unit = n_in * KB;
  const int o1 = n_in > 1 ? KB : 0, o2 = n_in > 2 ? 2 * KB : 0;  // missing streams re-read stream 0 with weight 0
  const B32 w1 = aie::broadcast<int16_t, 32>((int16_t)(n_in > 1 ? 0x3F80 : 0)).template cast_to<bfloat16>();
  const B32 w2 = aie::broadcast<int16_t, 32>((int16_t)(n_in > 2 ? 0x3F80 : 0)).template cast_to<bfloat16>();
  for (int rb = 0; rb < RBN; rb++) {
    A32 s0 = aie::zeros<accfloat, 32>(), s1 = s0, q0 = s0, q1 = s0;
    for (int q = 0; q < QS; q++) {
      in_need(in_ring, unit);
      const int ib = g_in_b;
      bfloat16 *st = g_stage + q * 512;
      for (int cb = 0; cb < 8; cb++) {
        const bfloat16 *pa = ring_r(in_ring, ib, cb * 128), *pb = ring_r(in_ring, ib, o1 + cb * 128), *pc = ring_r(in_ring, ib, o2 + cb * 128);
        A32 a0 = aie::mul(aie::load_v<32>(pa), ones), a1 = aie::mul(aie::load_v<32>(pa + 32), ones);
        a0 = aie::mac(a0, aie::load_v<32>(pb), w1);
        a1 = aie::mac(a1, aie::load_v<32>(pb + 32), w1);
        a0 = aie::mac(a0, aie::load_v<32>(pc), w2);
        a1 = aie::mac(a1, aie::load_v<32>(pc + 32), w2);
        B32 x0 = BF(a0), x1 = BF(a1);
        aie::store_v(st + cb * 64, x0);
        aie::store_v(st + cb * 64 + 32, x1);
        if constexpr (LN) {
          s0 = aie::mac(s0, x0, ones);
          s1 = aie::mac(s1, x1, ones);
          q0 = aie::mac(q0, x0, x0);  // exact fp32 squares of bf16 values
          q1 = aie::mac(q1, x1, x1);
        } else {
          q0 = aie::mac(q0, BF(aie::mul(x0, x0)), ones);
          q1 = aie::mac(q1, BF(aie::mul(x1, x1)), ones);
        }
      }
      in_done(in_ring, unit);
    }
    B32 r0, r1, rl0, rl1;
    A32 nb0, nb1;
    if constexpr (LN) {
      ln_stats(rowsum8(s0), rowsum8(q0), eps, r0, rl0, nb0);
      ln_stats(rowsum8(s1), rowsum8(q1), eps, r1, rl1, nb1);
    } else {
      r0 = rms_rstd(rowsum8(q0), eps);
      r1 = rms_rstd(rowsum8(q1), eps);
    }
    for (int q = 0; q < QS; q++) {
      out_begin(out_ring, UNIT_B);
      const int ob = g_out_b;
      bfloat16 *st = g_stage + q * 512;
      {
        OutS ps(g_panels);
        for (int cb = 0; cb < 8; cb++) {
          B32 x0 = aie::load_v<32>(st + cb * 64), x1 = aie::load_v<32>(st + cb * 64 + 32);
          aie::store_v(ring_w(out_ring, ob, cb * 128), x0);
          aie::store_v(ring_w(out_ring, ob, cb * 128 + 64), x1);
          B32 z0, z1;
          if constexpr (LN) {
            B32 ga = rep4(g_p0 + q * 64 + cb * 8), be = rep4(g_p1 + q * 64 + cb * 8);
            A32 t0 = nb0, t1 = nb1;
            t0 = aie::mac(t0, x0, r0);
            t0 = aie::mac(t0, x0, rl0);
            t1 = aie::mac(t1, x1, r1);
            t1 = aie::mac(t1, x1, rl1);
            B32 th, tl;
            split32(t0, th, tl);
            A32 y0 = aie::mul(be, ones);
            y0 = aie::mac(y0, th, ga);
            y0 = aie::mac(y0, tl, ga);
            split32(t1, th, tl);
            A32 y1 = aie::mul(be, ones);
            y1 = aie::mac(y1, th, ga);
            y1 = aie::mac(y1, tl, ga);
            z0 = BF(y0);
            z1 = BF(y1);
            aie::store_v(st + cb * 64, z0);  // for to_x
            aie::store_v(st + cb * 64 + 32, z1);
          } else {
            B32 sc = rep4(g_p0 + q * 64 + cb * 8);
            z0 = BF(aie::mul(BF(aie::mul(x0, r0)), sc));
            z1 = BF(aie::mul(BF(aie::mul(x1, r1)), sc));
          }
          ps.push(A64(aie::concat(z0, z1)).template to_vector<bfp16ebs8>());
        }
      }
      for (int i = 0; i < PANB / 64; i++) aie::store_v(ring_w(out_ring, ob, KB + 64 * i), aie::load_v<32>((const bfloat16 *)g_panels + 32 * i));
      if (to_x)
        for (int i = 0; i < KB / 64; i++) aie::store_v(ring_w(out_ring, ob, 64 * i), aie::load_v<32>(st + 32 * i));
      out_end(out_ring, UNIT_B);
    }
  }
}

extern "C" {

// X' = bf16(sum of n_in Q16 streams); mode 1: RMSNorm (torch bf16 numerics) -> panels; 2: LayerNorm -> panels;
// 3: LayerNorm -> panels and the X' part (the encoder norm). Params: a header slot (word 0 = the number of input
// streams, word 1 = eps as bf16 bits) + 2 slots (mode 1) or 4 (2/3).
void ew_norm(uint32_t *in_ring, uint32_t *out_ring, int32_t mode) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int pbytes = mode >= 2 ? 5 * SLOT_B : 3 * SLOT_B;  // [header slot][scale | gamma][beta]
  in_need(in_ring, pbytes);
  const int n_in = *(const int32_t *)in_at(in_ring, 0);
  const B32 eps = aie::broadcast<int16_t, 32>(*(const int16_t *)in_at(in_ring, 4)).template cast_to<bfloat16>();
  in_copy(in_ring, g_p0, SLOT_B, D * 2);
  if (mode >= 2) in_copy(in_ring, g_p1, 3 * SLOT_B, D * 2);
  in_done(in_ring, pbytes);
  if (mode >= 2)
    norm_rows<true>(in_ring, out_ring, n_in, mode == 3, eps);
  else
    norm_rows<false>(in_ring, out_ring, n_in, 0, eps);
}

// [8 rows x 64 nt] bfp panels (apan) times `nt` stacked 64x64 weight tiles -> 64 output columns, bf16 into dst
// [cb][8][8]. two_round: bf16(bf16(acc) + bias) (torch's einsum then + b); else bf16(acc + bias). btab: the bias per
// column block broadcast to [8 rows][8] (bias_tab).
static __attribute__((noinline)) void gemm8(const bfp16ebs8 *apan, const bfp16ebs8 *wt, int nt, const B64 *btab, bfloat16 *dst, int two_round) {
  const B64 ones64 = aie::broadcast<bfloat16, 64>(1.0f);
  for (int np = 0; np < 4; np++) {
    MM acc0, acc1;
    InS<aie_dm_resource::a> as(apan);
    for (int t = 0; t < nt; t++) {
      InS<aie_dm_resource::b> bs(wt + t * 4608 + np * 1152);
      for (int kk = 0; kk < 8; kk++) {
        BP a = as.pop();
        BP b0 = bs.pop(), b1 = bs.pop();
        if (t == 0 && kk == 0) {
          acc0.mul(a, aie::op_transpose(b0));
          acc1.mul(a, aie::op_transpose(b1));
        } else {
          acc0.mac(a, aie::op_transpose(b0));
          acc1.mac(a, aie::op_transpose(b1));
        }
      }
    }
    for (int j = 0; j < 2; j++) {
      const int cb = 2 * np + j;
      A64 r = j == 0 ? acc0.to_accum() : acc1.to_accum();
      B64 bb = btab[cb];
      if (two_round) r = aie::mul(BF(r), ones64);
      aie::store_v(dst + cb * 64, BF(aie::mac(r, bb, ones64)));
    }
  }
}

// ---- patch embedding: per k-step q: [3 weight tiles][bias 64] then 18 chunks of [24 col-blocks][8][8] bf16 -> 1024 B
static bfp16ebs8 *const g_wt = (bfp16ebs8 *)g_arena;
static bfp16ebs8 *const g_apan = (bfp16ebs8 *)(g_arena + 13824);
static bfloat16 *const g_pbias = (bfloat16 *)(g_arena + 15552);
static B64 *const g_ptab = (B64 *)(g_arena + 15680);  // 8 x 128 B

// 64 per-column values -> [cb][8 rows][8] (one B64 per column block)
static __attribute__((noinline, minsize)) void bias_tab(const bfloat16 *v64, B64 *tab) {
  for (int cb = 0; cb < 8; cb++) {
    B32 r = rep4(v64 + cb * 8);
    tab[cb] = aie::concat(r, r);
  }
}

__attribute__((minsize)) void ew_patch(uint32_t *in_ring, uint32_t *out_ring) {  // once per step: size over speed
  aie::set_rounding(aie::rounding_mode::conv_even);
  const B32 ones = aie::broadcast<bfloat16, 32>(1.0f);
  for (int q = 0; q < QS; q++) {
    in_need(in_ring, PATCH_PARAM_B);
    in_copy(in_ring, g_wt, 0, 3 * 4608);
    in_copy(in_ring, g_pbias, 3 * 4608, 128);  // the 64 bias values of this k-step
    in_done(in_ring, PATCH_PARAM_B);
    bias_tab(g_pbias, g_ptab);
    for (int rb = 0; rb < RBN; rb++) {
      in_need(in_ring, 3072);
      {  // the 8 x 192 bf16 chunk -> 24 bfp panels
        OutS as(g_apan);
        for (int cb = 0; cb < 24; cb++) {
          const bfloat16 *p = in_at(in_ring, cb * 128);
          as.push(A64(aie::concat(aie::load_v<32>(p), aie::load_v<32>(p + 32))).template to_vector<bfp16ebs8>());
        }
      }
      in_done(in_ring, 3072);
      alignas(64) bfloat16 obuf[512];
      gemm8(g_apan, g_wt, 3, g_ptab, obuf, 0);
      out_bytes(out_ring, obuf, KB);
    }
  }
}

// ---- the recurrent block's elementwise part, head (k-step) major -------------------------------------------
// Per head q: params (SSM_PARAM_B: header words [first, head]; bf16 vectors at element 64 sp = softplus(a_param),
// 128 conv_b, 192 + 64 t conv_w[t] (t = 0 oldest .. 3 current), 448 bx, 512 ba; then Wx, Wa as 64x64 weight
// tiles) and 18 chunks of 7 x 1024 B: y, x, x-1, x-2, x-3 (bf16 [cb 8][8 rows][8]), h rows 0-3, h rows 4-7
// (fp32 [cb 8][4 rows][8]). Output per chunk: [x][x-1][x-2][h rows 0-3][h rows 4-7][z panels 576 + 448 pad].
// torch's bf16 numerics are mirrored op by op (see pack8.ssm_ref).
static bfp16ebs8 *const g_wx = (bfp16ebs8 *)g_arena;
static bfp16ebs8 *const g_wa = (bfp16ebs8 *)(g_arena + 4608);
static bfloat16 *const g_sv = (bfloat16 *)(g_arena + 9216);       // the 768 bf16 parameter vector (1536 B)
static bfp16ebs8 *const g_xcp = (bfp16ebs8 *)(g_arena + 10752);   // xc as 8 bfp panels (576 B)
static bfloat16 *const g_xc = (bfloat16 *)(g_arena + 11328);      // xc bf16 [cb][8][8] (1024 B)
static bfloat16 *const g_gx = (bfloat16 *)(g_arena + 12352);      // gate_x pre-activation -> gate (1024 B)
static bfloat16 *const g_ga = (bfloat16 *)(g_arena + 13376);      // gate_a pre-activation -> gate -> a (1024 B, follows g_gx)
static float *const g_hout = (float *)(g_arena + 14400);          // h out fp32 [2 halves][cb 8][4 rows][8] (2048 B)
#ifdef SSM_LUT  // round 9: bf16(h) over gate_x (the recurrence reads gx[i] before it writes hb[i]), 1 - e2 over the
// parameter vector (dead once ssm_tables ran): g_tba takes the old g_hb
// slot: the arena ends at 21,568 and the sigmoid table's ab copy fits beside it
static bfloat16 *const g_hb = (bfloat16 *)(g_arena + 12352);      // bf16(h) [cb][8][8] (1024 B, = g_gx)
#else
static bfloat16 *const g_hb = (bfloat16 *)(g_arena + 16448);      // bf16(h) [cb][8][8] (1024 B)
#endif
// per-head tables, [cb] -> the 8 per-column values broadcast over 4 rows (B32) or 8 rows (B64)
static B32 *const g_tcw = (B32 *)(g_arena + 17472);               // conv taps [t 4][cb 8] (2048 B)
static B32 *const g_tcb = (B32 *)(g_arena + 19520);               // conv bias [cb] (512 B)
static B32 *const g_tsp = (B32 *)(g_arena + 20032);               // softplus(a_param) [cb] (512 B)
static B64 *const g_tbx = (B64 *)(g_arena + 20544);               // input-gate bias [cb] (1024 B)
#ifdef SSM_LUT
static B64 *const g_tba = (B64 *)(g_arena + 16448);               // a-gate bias [cb] (1024 B, the old g_hb slot)
#else
static B64 *const g_tba = (B64 *)(g_arena + 21568);               // a-gate bias [cb] (1024 B)
#endif
#if defined(SSM_EXP3)  // 10240: g_sv's tail + g_xcp (dead after the gate GEMMs); 9216 is a scratch slot of the exp passes
static bfloat16 *const g_e2 = (bfloat16 *)(g_arena + 10240);      // exp(2 log_a) -> sqrt(1 - .) (1024 B)
static_assert(20544 + 1024 <= sizeof(g_arena));
#elif defined(SSM_LUT)
static bfloat16 *const g_e2 = (bfloat16 *)(g_arena + 9216);       // exp(2 log_a) -> sqrt(1 - .) (1024 B, = g_sv)
static_assert(20544 + 1024 <= sizeof(g_arena));  // g_tbx is the last buffer now
#else
static bfloat16 *const g_e2 = (bfloat16 *)(g_arena + 22592);      // exp(2 log_a) -> sqrt(1 - .) (1024 B)
static_assert(22592 + 1024 <= sizeof(g_arena));
#endif

// the SSM's 6 KB output per chunk is exactly 8 output slots and always starts on a slot: reserve them once, store
// the 96 vectors straight into the ring, release them once
static inline bfloat16 *out_at(uint32_t *ring, int off) {
  int p = g_out_b + off;
  if (p >= OUT_B) p -= OUT_B;
  return (bfloat16 *)((char *)ring + p);
}

// round 8 (the SSM was compute-bound at 11,200 cycles per chunk, r8/ewprof.py):
// a 1 KB part of the unread input as a contiguous pointer; the one a ring wrap splits is copied to g_ptmp
alignas(64) static bfloat16 g_ptmp[KB / 2];
static __attribute__((noinline, minsize)) const bfloat16 *in_part(const uint32_t *ring, int off) {  // minsize: program memory
  int p = g_in_b + off;
  if (p >= IN_B) p -= IN_B;
  const bfloat16 *src = (const bfloat16 *)((const char *)ring + p);
  if (p + KB <= IN_B) return src;
  const int n1 = (IN_B - p) / 64;  // vectors before the wrap (IN_B % 64 == 0)
  for (int i = 0; i < n1; i++) aie::store_v(g_ptmp + 32 * i, aie::load_v<32>(src + 32 * i));
  for (int i = n1; i < KB / 64; i++) aie::store_v(g_ptmp + 32 * i, aie::load_v<32>((const bfloat16 *)ring + 32 * (i - n1)));
  return g_ptmp;
}
// nvec 64 B vectors from src to byte `off` of the current output position: at most one ring wrap, so two straight
// copy loops (the old per-vector out_at / in_at ran ~11 cycles per vector)
static __attribute__((noinline)) void copy_v(bfloat16 *__restrict d, const bfloat16 *__restrict s, int nvec) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (int i = 0; i < nvec; i += 2) {  // nvec even; pointer bumps (indexed addressing cost 8 bundles per pair)
    const B32 a = aie::load_v<32>(s), b = aie::load_v<32>(s + 32);
    s += 64;
    aie::store_v(d, a);
    aie::store_v(d + 32, b);
    d += 64;
  }
}
static __attribute__((noinline)) void out_copy(uint32_t *ring, int off, const bfloat16 *__restrict src, int nvec) {
  int p = g_out_b + off;
  if (p >= OUT_B) p -= OUT_B;
  int n1 = (OUT_B - p) / 64;  // the wrap is on a slot boundary (12 vectors): n1 is even
  if (n1 > nvec) n1 = nvec;
  copy_v((bfloat16 *)((char *)ring + p), src, n1);
  if (n1 < nvec) copy_v((bfloat16 *)ring, src + 32 * n1, nvec - n1);
}
// the SSM's gate GEMM: gemm8 for one weight tile with its two-rounding epilogue, constant trip counts (gemm8's runtime
// nt kept its k loop a 24-bundle branch loop: ~1,190 cycles per call). The same MAC order: bit-identical.
static __attribute__((noinline)) void gate_gemm(const bfp16ebs8 *apan, const bfp16ebs8 *wt, const B64 *__restrict btab,
                                                bfloat16 *__restrict dst) {
  const B64 ones64 = aie::broadcast<bfloat16, 64>(1.0f);
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int np = 0; np < 4; np++) {
    MM acc0, acc1;
    InS<aie_dm_resource::a> as(apan);
    InS<aie_dm_resource::b> bs(wt + np * 1152);
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
      BP a = as.pop();
      BP b0 = bs.pop(), b1 = bs.pop();
      if (kk == 0) {
        acc0.mul(a, aie::op_transpose(b0));
        acc1.mul(a, aie::op_transpose(b1));
      } else {
        acc0.mac(a, aie::op_transpose(b0));
        acc1.mac(a, aie::op_transpose(b1));
      }
    }
    const A64 r0 = aie::mul(BF(acc0.to_accum()), ones64), r1 = aie::mul(BF(acc1.to_accum()), ones64);
    aie::store_v(dst + np * 128, BF(aie::mac(r0, btab[2 * np], ones64)));
    aie::store_v(dst + np * 128 + 64, BF(aie::mac(r1, btab[2 * np + 1], ones64)));
  }
}

static __attribute__((noinline, minsize)) void ssm_tables() {  // once per head
  for (int cb = 0; cb < 8; cb++) {
    for (int t = 0; t < 4; t++) g_tcw[t * 8 + cb] = rep4(g_sv + 192 + 64 * t + cb * 8);
    g_tcb[cb] = rep4(g_sv + 128 + cb * 8);
    g_tsp[cb] = rep4(g_sv + 64 + cb * 8);
  }
  bias_tab(g_sv + 448, g_tbx);
  bias_tab(g_sv + 512, g_tba);
}

void ew_ssm(uint32_t *in_ring, uint32_t *out_ring) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const nk8::ExpC kc;
  const B32 ones = aie::broadcast<bfloat16, 32>(1.0f);
#ifdef PROF
  for (int i = 0; i < 12; i++) g_ph[i] = 0;
  g_nchunk = 0;
  g_tlast = stamp();
#endif
  for (int q = 0; q < QS; q++) {
    in_need(in_ring, SSM_PARAM_B);
    PH(10);
    const int first = *(const int32_t *)in_at(in_ring, 0);
    in_copy(in_ring, g_sv, 0, 1536);
    in_copy(in_ring, g_wx, 1536, 4608);
    in_copy(in_ring, g_wa, 1536 + 4608, 4608);
    in_done(in_ring, SSM_PARAM_B);
    const B32 mfirst = aie::broadcast<bfloat16, 32>(first ? 1.0f : 0.0f);
    const B32 mrest = aie::broadcast<bfloat16, 32>(first ? 0.0f : 1.0f);
    ssm_tables();
    PH(11);
    for (int rb = 0; rb < RBN; rb++) {
      in_need(in_ring, 7 * KB);
      PH(0);
      // round 8: contiguous views of the chunk's 7 input parts (a part the ring wrap splits is copied to g_ptmp once);
      // the loops below use plain pointers instead of a wrap check per vector (in_at)
      const bfloat16 *P[7];
      for (int p = 0; p < 7; p++) P[p] = in_part(in_ring, p * KB);
      // conv1d: xc = bf16(bf16(sum_t bf16(x_t w_t)) + b), taps x-3 (part 4) .. x (part 1)
      AIE_LOOP_MIN_ITERATION_COUNT(8)
      for (int cb = 0; cb < 8; cb++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int off = cb * 64 + h * 32;
          A32 s = aie::mul(BF(aie::mul(aie::load_v<32>(P[4] + off), g_tcw[0 * 8 + cb])), ones);
          s = aie::mac(s, BF(aie::mul(aie::load_v<32>(P[3] + off), g_tcw[1 * 8 + cb])), ones);
          s = aie::mac(s, BF(aie::mul(aie::load_v<32>(P[2] + off), g_tcw[2 * 8 + cb])), ones);
          s = aie::mac(s, BF(aie::mul(aie::load_v<32>(P[1] + off), g_tcw[3 * 8 + cb])), ones);
          aie::store_v(g_xc + off, BF(aie::mac(A32(aie::mul(BF(s), ones)), g_tcb[cb], ones)));
        }
      }
      PH(1);
      {  // xc -> bfp panels for the gate GEMMs
        OutS ps(g_xcp);
        for (int cb = 0; cb < 8; cb++) ps.push(A64(aie::load_v<64>(g_xc + cb * 64)).template to_vector<bfp16ebs8>());
      }
      gate_gemm(g_xcp, g_wx, g_tbx, g_gx);
      gate_gemm(g_xcp, g_wa, g_tba, g_ga);
      PH(2);
#ifndef SSM_NO_GATES
#ifdef SSM_LUT
      ssm9::sigmoid_lut_n(g_gx, 32);  // gate_x and gate_a (contiguous)
#else
      nk8::sigmoid_n(g_gx, 32, kc);  // gate_x and gate_a (contiguous)
#endif
      PH(3);
#ifdef SSM_EXP3  // round 9: the same ops as pipelined passes (s1 = 9216, s2 = g_e2, s3 / s4 = g_hout, all free here)
      ssm9::loga_exp_a2(g_ga, g_tsp, (bfloat16 *)(g_arena + 9216), g_e2, (bfloat16 *)g_hout, (bfloat16 *)g_hout + 512);
#else
      AIE_LOOP_MIN_ITERATION_COUNT(8)
      for (int i = 0; i < 16; i++)  // log_a = bf16(bf16(-8 gate_a) sp)
        aie::store_v(g_ga + i * 32, BF(aie::mul(BF(aie::mul(aie::load_v<32>(g_ga + i * 32), (bfloat16)-8.0f)), g_tsp[i / 2])));
      nk8::exp_a2_n(g_ga, g_e2, 16, kc);  // a, 1 - e2 (e2 = exp(2 log_a) = a^2: one exp instead of two)
#endif
      PH(4);
#ifdef SSM_LUT
      ssm9::sqrt_lut_n(g_e2, 16);  // mult = sqrt(1 - e2), exact
#else
      nk8::sqrt_n(g_e2, 16, kc);  // mult = sqrt(1 - e2)
#endif
      PH(5);
#endif
      AIE_LOOP_MIN_ITERATION_COUNT(8)
      for (int cb = 0; cb < 8; cb++) {
#pragma unroll
        for (int half = 0; half < 2; half++) {
          const int i = 2 * cb + half;
          B32 xc = aie::load_v<32>(g_xc + i * 32);
          B32 gx = aie::load_v<32>(g_gx + i * 32), a = aie::load_v<32>(g_ga + i * 32), mult = aie::load_v<32>(g_e2 + i * 32);
          B32 gated = BF(aie::mul(xc, gx));
          B32 nx = BF(aie::mul(gated, mult));
          // h = a h0 + nx (fp32; first frame: h = gated), hb = bf16(h)
          A32 h0a(aie::load_v<32>((const float *)P[5 + half] + cb * 32));
          B32 h0h, h0l;
          nk8::split(h0a, h0h, h0l);
          A32 h = aie::mul(a, h0h);
          h = aie::mac(h, a, h0l);
          h = aie::mac(h, nx, ones);
          B32 hh, hl;
          nk8::split(h, hh, hl);
          A32 hsel = aie::mul(hh, mrest);  // first ? gated : h
          hsel = aie::mac(hsel, hl, mrest);
          hsel = aie::mac(hsel, gated, mfirst);
          aie::store_v(g_hout + half * 256 + cb * 32, FV(hsel));
          aie::store_v(g_hb + i * 32, BF(hsel));
        }
      }
      PH(6);
      {
        OutS zs(g_panels);
        for (int cb = 0; cb < 8; cb++) {  // z = bf16(hb * y) -> panels
          B64 y = aie::load_v<64>(P[0] + cb * 64);
          B64 z = BF(aie::mul(aie::load_v<64>(g_hb + cb * 64), y));
          zs.push(A64(z).template to_vector<bfp16ebs8>());
        }
      }
      PH(7);
      // output (8 slots): x, x-1, x-2 (the conv cache shifts by one frame), h rows 0-3, h rows 4-7, z panels + pad
      lacq(out_ring, L_OUT_FREE, 8);
      PH(8);
      for (int p = 0; p < 3; p++) out_copy(out_ring, p * KB, P[1 + p], KB / 64);
      out_copy(out_ring, 3 * KB, (const bfloat16 *)g_hout, 2 * KB / 64);
      out_copy(out_ring, 5 * KB, (const bfloat16 *)g_panels, KB / 64);  // panels 576 B + g_panels' zero tail as the pad
#ifdef PROF  // record in the pad (unused bytes of the output): magic, chunks so far, phase totals (r8/ewprof.py)
      {
        uint32_t *rec = (uint32_t *)out_at(out_ring, 5 * KB + PANB);
        rec[0] = 0x504D5353u;
        rec[1] = ++g_nchunk;
        for (int i = 0; i < 12; i++) rec[2 + i] = g_ph[i];
      }
#endif
      lrel(out_ring, L_OUT_FILL, 8);
      g_out_b += 6 * KB;
      if (g_out_b >= OUT_B) g_out_b -= OUT_B;
      in_done(in_ring, 7 * KB);
      PH(9);
    }
  }
}

// The EW core's only entry. Every op's first params slot names it (header word 3: 1 rms, 2 ln, 3 ln_out, 4 ssm), so
// the core's control code is one call in IRON's worker loop, independent of the op list and of the depth.
void ew_op(uint32_t *in_ring, uint32_t *out_ring) {
  in_need(in_ring, 16);
  const int kind = *(const int32_t *)in_at(in_ring, 12);
#ifdef EW_NO_NORM  // profiling builds of SSM-only tests (t3): the norm code out of program memory
  (void)kind;
  ew_ssm(in_ring, out_ring);
#else
  if (kind == 4)
    ew_ssm(in_ring, out_ring);
  else
    ew_norm(in_ring, out_ring, kind);
#endif
}
}
