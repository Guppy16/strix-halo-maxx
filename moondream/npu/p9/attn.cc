// Fused attention for one (head, 96-query tile) item on one core, bf16emu.py numerics:
//   s = bf(bf(Q K^T) * scale);  p = bf(exp(bf(s - rowmax)));  p = bf(p * bf(1/rowsum_fp32));  o = bf(P V)
// Two passes over the keys in 32-key chunks: pass 0 row max (bf16, exact), pass 1 e = exp(s - max),
// fp32 row sum and fp32 PV accumulation; the bf16 1/sum is applied once at the end. QK^T and PV use the native bf16 mmul (8x8x8 panels, fp32 accumulate);
// Q / K / V panels are the qkv GEMM's blocked bf16 tiles as they are (8x8 blocks, 72 = 9 blocks per head).
//
// Buffers (bf16, 8x8 blocks row-major inside, block-row major):
//   q:  [12 row blocks][9 dim blocks][64]           96 x 72
//   kc: [12 key blocks][9 dim blocks][64]           96 keys x 72 (one chunk)   -> B panels need a transpose
//   vc: [12 key blocks][9 dim blocks][64]           96 keys x 72 (one chunk)   -> B panels as stored
//   o:  [12 row blocks][9 dim blocks][64]           96 x 72, bf16 out
#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

constexpr int QT = 64, HD = 72, KC = 32;  // query tile, head dim, keys per chunk
constexpr int NB = HD / 8;                // 9 dim blocks
constexpr int RB = QT / 8, CB = KC / 8;   // 8 row blocks, 4 key blocks
constexpr int NKEYS = ATTN_KEYS;          // 729 real keys (the last chunk is padded with zero keys)
constexpr int NCH = (NKEYS + KC - 1) / KC;

using MM = aie::mmul<8, 8, 8, bfloat16, bfloat16>;
using B32 = aie::vector<bfloat16, 32>;
using B64 = aie::vector<bfloat16, 64>;
using A32 = aie::accum<accfloat, 32>;
using A64 = aie::accum<accfloat, 64>;
#define BF(e) ((e).template to_vector<bfloat16>())

alignas(64) static bfloat16 s_chunk[QT * KC];   // bf16 scores of the current chunk, [rb][cb][64]
alignas(64) static bfloat16 g_max[QT];           // row max (bf16, as the reference)
alignas(64) static bfloat16 mask_full[CB * 64], neg_full[CB * 64], mask_last[CB * 64], neg_last[CB * 64];
alignas(64) static float g_maxv[QT], g_sumv[QT];  // row max / sum as 8-lane fp32 vectors per row block
alignas(64) static bfloat16 g_rinv[QT];          // bf16(1 / sum)
alignas(64) static float o_acc[QT * HD];         // fp32 PV accumulator [row block][dim block][64]
static bfloat16 g_scale;
static int g_pass, g_chunk;


#include "expsplit.h"  // exp_acc(A32) -> B32, recip_b(B32) -> B32 (shared with the GELU epilogue)

extern "C" {

void attn_begin(bfloat16 *q) {  // per item: reset the running state; q stays resident for both passes
  for (int r = 0; r < QT; r++) { g_max[r] = (bfloat16)(-1e30f); g_maxv[r] = -1e30f; g_sumv[r] = 0.0f; }
  const int valid_last = NKEYS - (NCH - 1) * KC;
  for (int cb = 0; cb < CB; cb++)
    for (int i = 0; i < 8; i++)
      for (int l = 0; l < 8; l++) {
        const int idx = cb * 64 + i * 8 + l;
        const bool ok = cb * 8 + l < valid_last;
        mask_full[idx] = (bfloat16)1.0f; neg_full[idx] = (bfloat16)0.0f;
        mask_last[idx] = ok ? (bfloat16)1.0f : (bfloat16)0.0f;
        neg_last[idx] = ok ? (bfloat16)0.0f : (bfloat16)(-1e30f);
      }
  for (int i = 0; i < QT * HD / 32; i++) aie::store_v(o_acc + 32 * i, aie::zeros<float, 32>());
  g_scale = (bfloat16)ATTN_SCALE;
  g_pass = 0; g_chunk = 0;
}

// out-of-line exp: two inlined 32-lane chains per block spilled 13 KB of stack
#ifdef EXP_FAST
// cheaper exp for the softmax: exact argument via split MACs, 2^f by a plain bf16 Horner (rounded per step,
// ~1 bf16 ulp), exponent assembled with int ops. ~3x fewer ops than exp_acc.
static __attribute__((noinline)) B32 exp_ni(A32 xa) {
  B32 xh, xl;
  split_acc(xa, xh, xl);
  const bfloat16 L_HI = 1.4426950408889634f;
  const bfloat16 L_LO = 1.4426950408889634f - (float)L_HI;
  A32 za = aie::mul(xh, L_HI);
  za = aie::mac(za, xh, L_LO);
  za = aie::mac(za, xl, L_HI);
  aie::vector<float, 32> z = za.template to_vector<float>();
  z = aie::max(aie::min(z, aie::broadcast<float, 32>(100.0f)), aie::broadcast<float, 32>(-100.0f));
  aie::vector<int32_t, 32> n = aie::to_fixed<int32_t>(aie::sub(z, 0.5f), 0);
  B32 f = aie::accum<accfloat, 32>(aie::sub(z, aie::to_float(n, 0))).template to_vector<bfloat16>();
  A32 p = aie::mul(f, bfloat16(0.05550411f));
  p = aie::mac(aie::accum<accfloat, 32>(aie::broadcast<float, 32>(0.24022651f)), f, BF(p));
  p = aie::mac(aie::accum<accfloat, 32>(aie::broadcast<float, 32>(0.69314718f)), f, BF(p));
  p = aie::mac(aie::accum<accfloat, 32>(aie::broadcast<float, 32>(1.0f)), f, BF(p));
  aie::vector<int32_t, 32> ebits = aie::add(aie::upshift(n, 23), p.template to_vector<float>().template cast_to<int32_t>());
  return aie::accum<accfloat, 32>(ebits.template cast_to<float>()).template to_vector<bfloat16>();
}
#else
static __attribute__((noinline)) B32 exp_ni(A32 a) { return exp_acc(a); }
#endif

// S chunk = bf(bf(Q Kc^T) * scale) into s_chunk, [rb][cb][64]; K blocks transposed in registers per use
// (a transposed scratch copy cost 4.6 KB of L1 the core does not have)
static inline void scores(const bfloat16 *q, const bfloat16 *kc) {
  for (int rb = 0; rb < RB; rb++) {
    for (int cb = 0; cb < CB; cb++) {
      MM acc;
#pragma unroll
      for (int j = 0; j < NB; j++) {
        B64 a = aie::load_v<64>(q + (rb * NB + j) * 64);
        B64 bt = aie::transpose(aie::load_v<64>(kc + (cb * NB + j) * 64), 8, 8);  // [8 keys][8 dims] -> [k][n]
        if (j == 0) acc.mul(a, bt); else acc.mac(a, bt);
      }
      B64 s = BF(acc);
      s = BF(aie::mul(s, g_scale));
      aie::store_v(s_chunk + (rb * CB + cb) * 64, s);
    }
  }
}

// lane i*8+l -> value[i] for 8 rows (bf16): eight 8-lane broadcasts
static inline B64 rows8(const bfloat16 *v8) {
  return aie::concat(aie::concat(aie::broadcast<bfloat16, 8>(v8[0]), aie::broadcast<bfloat16, 8>(v8[1]),
                                 aie::broadcast<bfloat16, 8>(v8[2]), aie::broadcast<bfloat16, 8>(v8[3])),
                     aie::concat(aie::broadcast<bfloat16, 8>(v8[4]), aie::broadcast<bfloat16, 8>(v8[5]),
                                 aie::broadcast<bfloat16, 8>(v8[6]), aie::broadcast<bfloat16, 8>(v8[7])));
}
// per-row (8 lanes) reductions of a 64-lane [8 rows][8 lanes] block without lane shuffles: transpose the
// block (verified by the K^T panels), then reduce the eight 8-lane extracts elementwise -> lane l = row l.
static __attribute__((noinline)) aie::vector<float, 8> rowmax8(B64 v) {
  B64 t = aie::transpose(v, 8, 8);  // t[i*8 + l] = v[l*8 + i]
  // three native 16-lane bf16 maxes (rows l and l+8 interleaved), then one 8-lane fp32 max
  aie::vector<bfloat16, 16> m = aie::max(t.template extract<16>(0), t.template extract<16>(1));
  m = aie::max(m, aie::max(t.template extract<16>(2), t.template extract<16>(3)));
  aie::vector<float, 16> mf = aie::accum<accfloat, 16>(m).template to_vector<float>();
  return aie::max(mf.template extract<8>(0), mf.template extract<8>(1));
}
static __attribute__((noinline)) aie::vector<float, 8> rowsum8(B64 e) {  // fp32 sum over the 8 lanes of each row
  B64 t = aie::transpose(e, 8, 8);
  const aie::vector<bfloat16, 16> ones = aie::broadcast<bfloat16, 16>(1.0f);
  aie::accum<accfloat, 16> a = aie::mul(t.template extract<16>(0), ones);  // exact: x * 1 accumulated in fp32
  for (int i = 1; i < 4; i++) a = aie::mac(a, t.template extract<16>(i), ones);
  aie::vector<float, 16> af = a.template to_vector<float>();
  return aie::add(af.template extract<8>(0), af.template extract<8>(1));
}

// one key chunk of the current pass. Pass 0: row max. Pass 1: e = bf(exp(bf(s - max))), fp32 row sum,
// O_acc += e V (fp32). The bf16 1/sum is applied once at the end (attn_end).
void attn_chunk(bfloat16 *q, bfloat16 *kc, bfloat16 *vc) {
  event0();
  const bool last = g_chunk == NCH - 1;
  const bfloat16 *maskv = last ? mask_last : mask_full;
  const bfloat16 *negv = last ? neg_last : neg_full;
#ifndef PROF_NOSCORES
  scores(q, kc);  // PROF_NOSCORES: timing-only variant reuses the previous chunk's scores
#endif
  if (g_pass == 0) {
    for (int rb = 0; rb < RB; rb++) {
      B64 m = aie::load_v<64>(s_chunk + (rb * CB) * 64);
      m = aie::add(BF(aie::mul(m, aie::load_v<64>(maskv))), aie::load_v<64>(negv));  // padded keys -> -1e30
      for (int cb = 1; cb < CB; cb++) {
        B64 t = aie::load_v<64>(s_chunk + (rb * CB + cb) * 64);
        t = aie::add(BF(aie::mul(t, aie::load_v<64>(maskv + cb * 64))), aie::load_v<64>(negv + cb * 64));
        m = aie::max(m, t);
      }
      aie::vector<float, 8> mv = aie::max(rowmax8(m), aie::load_v<8>(g_maxv + rb * 8));
      aie::store_v(g_maxv + rb * 8, mv);
      aie::store_v(g_max + rb * 8, aie::accum<accfloat, 8>(mv).template to_vector<bfloat16>());
    }
  } else {
    for (int rb = 0; rb < RB; rb++) {
      B64 mrow = rows8(g_max + rb * 8);
      B64 e[CB];
      aie::vector<float, 8> rs = aie::load_v<8>(g_sumv + rb * 8);
      for (int cb = 0; cb < CB; cb++) {
        B64 d = aie::sub(aie::load_v<64>(s_chunk + (rb * CB + cb) * 64), mrow);
#ifdef PROF_NOEXP
        B32 e0 = d.template extract<32>(0), e1 = d.template extract<32>(1);  // timing only
#else
        B32 e0 = exp_ni(aie::accum<accfloat, 32>(d.template extract<32>(0)));
        B32 e1 = exp_ni(aie::accum<accfloat, 32>(d.template extract<32>(1)));
#endif
        e[cb] = aie::concat(e0, e1);
        e[cb] = BF(aie::mul(e[cb], aie::load_v<64>(maskv + cb * 64)));
        rs = aie::add(rs, rowsum8(e[cb]));
      }
      aie::store_v(g_sumv + rb * 8, rs);
#ifndef PROF_NOPV
      for (int j = 0; j < NB; j++) {
        float *oc = o_acc + (rb * NB + j) * 64;
        MM acc((A64(aie::load_v<64>(oc))));
#pragma unroll
        for (int cb = 0; cb < CB; cb++) acc.mac(e[cb], aie::load_v<64>(vc + (cb * NB + j) * 64));
        aie::store_v(oc, acc.template to_vector<float>());
      }
#endif
    }
  }
  if (++g_chunk == NCH) { g_chunk = 0; g_pass++; }
  event1();
}

void attn_copyq(bfloat16 *q, bfloat16 *a, bfloat16 *b) {  // two 32x72 chunk objects -> the resident 64 x 72 Q tile
  for (int i = 0; i < KC * HD / 32; i++) {
    aie::store_v(q + 32 * i, aie::load_v<32>(a + 32 * i));
    aie::store_v(q + KC * HD + 32 * i, aie::load_v<32>(b + 32 * i));
  }
}

static inline void end_half(bfloat16 *o, int h) {  // row blocks 4h..4h+3: o = bf(O_acc * bf(1/sum)), fp32 product via hi/lo
  for (int rb = 4 * h; rb < 4 * h + 4; rb++) {
    B64 rrow = rows8(g_rinv + rb * 8);
    for (int j = 0; j < NB; j++) {
      A64 acc(aie::load_v<64>(o_acc + (rb * NB + j) * 64));
#ifdef SIMPLE_NORM
      aie::store_v(o + ((rb - 4 * h) * NB + j) * 64, BF(aie::mul(BF(acc), rrow)));
#else
      B64 hi = BF(acc);
      B64 lo = BF(aie::mac(acc, hi, (bfloat16)-1.0f));
      A64 prod = aie::mul(hi, rrow);
      prod = aie::mac(prod, lo, rrow);
      aie::store_v(o + ((rb - 4 * h) * NB + j) * 64, BF(prod));
#endif
    }
  }
}
#ifdef DEBUG_STATS
// debug: the output tile carries per row: dim 0 = bf16(row sum), dim 1 = row max, dim 2 = bf16(1/sum), rest 0
static void dbg_half(bfloat16 *o, int h) {
  for (int rb = 4 * h; rb < 4 * h + 4; rb++)
    for (int j = 0; j < NB; j++)
      for (int i = 0; i < 8; i++)
        for (int l = 0; l < 8; l++) {
          const int r = rb * 8 + i;
          bfloat16 v = (bfloat16)0.0f;
          if (j == 0 && l == 0) v = (bfloat16)g_sumv[r];
          if (j == 0 && l == 1) v = g_max[r];
          if (j == 0 && l == 2) v = g_rinv[r];
          o[((rb - 4 * h) * NB + j) * 64 + i * 8 + l] = v;
        }
}
void attn_end0(bfloat16 *o) {
  for (int r = 0; r < QT; r++) g_rinv[r] = (bfloat16)(1.0f / g_sumv[r]);
  dbg_half(o, 0);
}
void attn_end1(bfloat16 *o) { dbg_half(o, 1); }
#else
void attn_end0(bfloat16 *o) {
  for (int r = 0; r < QT; r++) g_rinv[r] = (bfloat16)(1.0f / g_sumv[r]);
  end_half(o, 0);
}
void attn_end1(bfloat16 *o) { end_half(o, 1); }
#endif
}
