// Fused attention v3 for one (head, 64-query tile) item per core: ONE pass over 64-key chunks with an online
// softmax, K^T panels supplied by the producer (no transposes), half of the fp32 O accumulator living in the
// output object, and an exp with no emulated fp32 ops. bf16emu.py numerics as far as the single pass allows:
//   s = bf(bf(Q K^T) * scale) per chunk; running row max m (bf16); e = bf(exp(bf(s - m)));
//   sum, O accumulate in fp32; when m grows by dm the running sum and O are rescaled by bf(exp(-dm)) (MAC-unit
//   hi/lo products, no vector selects: the rescale is a scalar branch per row block);
//   o = bf(O * bf(1/sum)) at the end.
// Buffers (bf16 8x8 blocks, block-row major): q [8 rb][9 j][64]; kt [8 cb][9 j][8 dims][8 keys] (K^T blocks);
// vc [8 cb][9 j][8 keys][8 dims]; o [8 rb][9 j][64].
#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

constexpr int QT = 48, HD = 72, KC = 48;  // 48-query tiles, 48-key chunks: Q, O and chunk objects are all 48 x 72
constexpr int NB = HD / 8, RB = QT / 8, CB = KC / 8;
constexpr int NKEYS = ATTN_KEYS;
constexpr int NCH = (NKEYS + KC - 1) / KC;  // 16

using MM = aie::mmul<8, 8, 8, bfloat16, bfloat16>;
using B32 = aie::vector<bfloat16, 32>;
using B64 = aie::vector<bfloat16, 64>;
using A32 = aie::accum<accfloat, 32>;
using A64 = aie::accum<accfloat, 64>;
#define BF(e) ((e).template to_vector<bfloat16>())

alignas(64) static float o_hi[(QT / 2) * HD];   // fp32 O rows 24..47; rows 0..23 live in the bf16 output object
alignas(64) static bfloat16 g_maxb[QT];          // running row max, bf16
alignas(64) static float g_maxv[QT], g_sumv[QT]; // running row max / sum as 8-lane fp32 vectors
alignas(64) static bfloat16 mask_full[64], neg_full[64], mask_last[CB * 64], neg_last[CB * 64];  // full-chunk masks: one row, stride 0
alignas(64) static bfloat16 s_rb[CB * 64];       // scores of one row block, [cb][64]
alignas(64) static bfloat16 e_rb[CB * 64];       // exp of one row block
static bfloat16 g_scale;
static int g_chunk;
static float *g_o_lo;  // fp32 O rows 0..31 = the output object (acquired for the whole item)

static inline float *o_rows(int rb) { return rb < RB / 2 ? g_o_lo + rb * NB * 64 : o_hi + (rb - RB / 2) * NB * 64; }

// exp(x) for an fp32 accumulator (|x| <= 60), bf16 result, no emulated fp32 ops: z = x log2e via split MACs,
// q = round(z * 256) (accumulator srs), n = q >> 8, f = (q & 255) / 256 as bf16, 2^f by a bf16 Horner,
// exponent assembled with int ops. 2^f error <= ln2/512 + bf16 rounding ~ 0.3%.
static __attribute__((noinline)) B32 exp_srs(A32 xa) {
  B32 xh = BF(xa);
  B32 xl = BF(aie::mac(xa, xh, (bfloat16)-1.0f));
  const bfloat16 L_HI = 1.4426950408889634f;
  const bfloat16 L_LO = 1.4426950408889634f - (float)L_HI;
  A32 za = aie::mul(xh, L_HI);
  za = aie::mac(za, xh, L_LO);
  za = aie::mac(za, xl, L_HI);
  aie::vector<int32_t, 32> qz = aie::to_fixed<int32_t>(za.template to_vector<float>(), 8);  // round(z * 256), conversion unit
  qz = aie::max(aie::min(qz, aie::broadcast<int32_t, 32>(60 * 256)), aie::broadcast<int32_t, 32>(-60 * 256));
  aie::vector<int32_t, 32> n = aie::downshift(qz, 8);                 // arithmetic shift = floor(z)
  aie::vector<int32_t, 32> fr = aie::sub(qz, aie::upshift(n, 8));      // 0..255
  B32 f = aie::accum<accfloat, 32>(aie::to_float(fr, 8)).template to_vector<bfloat16>();  // fr / 256 as bf16 (exact: 8 bits)
  A32 p = aie::mul(f, bfloat16(0.05550411f));
  p = aie::mac(aie::accum<accfloat, 32>(aie::broadcast<float, 32>(0.24022651f)), f, BF(p));
  p = aie::mac(aie::accum<accfloat, 32>(aie::broadcast<float, 32>(0.69314718f)), f, BF(p));
  p = aie::mac(aie::accum<accfloat, 32>(aie::broadcast<float, 32>(1.0f)), f, BF(p));
  aie::vector<int32_t, 32> ebits = aie::add(aie::upshift(n, 23), p.template to_vector<float>().template cast_to<int32_t>());
  return aie::accum<accfloat, 32>(ebits.template cast_to<float>()).template to_vector<bfloat16>();
}

static inline B64 rows8(const bfloat16 *v8) {
  return aie::concat(aie::concat(aie::broadcast<bfloat16, 8>(v8[0]), aie::broadcast<bfloat16, 8>(v8[1]),
                                 aie::broadcast<bfloat16, 8>(v8[2]), aie::broadcast<bfloat16, 8>(v8[3])),
                     aie::concat(aie::broadcast<bfloat16, 8>(v8[4]), aie::broadcast<bfloat16, 8>(v8[5]),
                                 aie::broadcast<bfloat16, 8>(v8[6]), aie::broadcast<bfloat16, 8>(v8[7])));
}
static __attribute__((noinline)) aie::vector<float, 8> rowmax8(B64 v) {
  B64 t = aie::transpose(v, 8, 8);
  aie::vector<bfloat16, 16> m = aie::max(t.template extract<16>(0), t.template extract<16>(1));
  m = aie::max(m, aie::max(t.template extract<16>(2), t.template extract<16>(3)));
  aie::vector<float, 16> mf = aie::accum<accfloat, 16>(m).template to_vector<float>();
  return aie::max(mf.template extract<8>(0), mf.template extract<8>(1));
}
static __attribute__((noinline)) aie::vector<float, 8> rowsum8(B64 e) {
  B64 t = aie::transpose(e, 8, 8);
  const aie::vector<bfloat16, 16> ones = aie::broadcast<bfloat16, 16>(1.0f);
  aie::accum<accfloat, 16> a = aie::mul(t.template extract<16>(0), ones);
  for (int i = 1; i < 4; i++) a = aie::mac(a, t.template extract<16>(i), ones);
  aie::vector<float, 16> af = a.template to_vector<float>();
  return aie::add(af.template extract<8>(0), af.template extract<8>(1));
}

extern "C" {

void attn_begin(bfloat16 *q, bfloat16 *o) {  // o: the output object, hosts fp32 O rows 0..31 during the item
  g_o_lo = (float *)o;
  for (int r = 0; r < QT; r++) { g_maxb[r] = (bfloat16)(-1e30f); g_maxv[r] = -1e30f; g_sumv[r] = 0.0f; }
  for (int i = 0; i < (QT / 2) * HD / 32; i++) { aie::store_v(g_o_lo + 32 * i, aie::zeros<float, 32>()); aie::store_v(o_hi + 32 * i, aie::zeros<float, 32>()); }
  const int valid_last = NKEYS - (NCH - 1) * KC;
  for (int cb = 0; cb < CB; cb++)
    for (int i = 0; i < 8; i++)
      for (int l = 0; l < 8; l++) {
        const int idx = cb * 64 + i * 8 + l;
        const bool ok = cb * 8 + l < valid_last;
        if (cb == 0) { mask_full[idx] = (bfloat16)1.0f; neg_full[idx] = (bfloat16)0.0f; }
        mask_last[idx] = ok ? (bfloat16)1.0f : (bfloat16)0.0f;
        neg_last[idx] = ok ? (bfloat16)0.0f : (bfloat16)(-1e30f);
      }
  g_scale = (bfloat16)ATTN_SCALE;
  g_chunk = 0;
}

// one 64-key chunk: for each row block: scores -> chunk max -> rescale if the running max grew -> e -> sum, PV
void attn_chunk(bfloat16 *q, bfloat16 *kt, bfloat16 *vc) {
  event0();
  const bool last = g_chunk == NCH - 1;
  const bfloat16 *maskv = last ? mask_last : mask_full;
  const bfloat16 *negv = last ? neg_last : neg_full;
  const int mstride = last ? 64 : 0;  // scalar: the full-chunk masks are one row reused for every key block
  for (int rb = 0; rb < RB; rb++) {
    // scores of this row block: s_rb[cb] = bf(bf(Q[rb] Kt[cb]) * scale), masked for the max
    B64 mx;
    for (int cb = 0; cb < CB; cb++) {
      MM acc;
#pragma unroll
      for (int j = 0; j < NB; j++) {
        B64 a = aie::load_v<64>(q + (rb * NB + j) * 64);
        B64 bt = aie::load_v<64>(kt + (cb * NB + j) * 64);
        if (j == 0) acc.mul(a, bt); else acc.mac(a, bt);
      }
      B64 s = BF(aie::mul(BF(acc), g_scale));
      aie::store_v(s_rb + cb * 64, s);
      B64 sm = aie::add(BF(aie::mul(s, aie::load_v<64>(maskv + cb * mstride))), aie::load_v<64>(negv + cb * mstride));
      mx = cb == 0 ? sm : aie::max(mx, sm);
    }
    // running max update; rescale sum / O by bf(exp(m_old - m_new)) where it grew (scalar branch per row block)
    aie::vector<float, 8> m_old = aie::load_v<8>(g_maxv + rb * 8);
    aie::vector<float, 8> m_new = aie::max(rowmax8(mx), m_old);
    aie::vector<float, 8> dm = aie::sub(m_old, m_new);  // <= 0
    if (aie::reduce_min(dm) < 0.0f) {
      aie::store_v(g_maxv + rb * 8, m_new);
      aie::store_v(g_maxb + rb * 8, aie::accum<accfloat, 8>(m_new).template to_vector<bfloat16>());
      // scale per row: exp(dm) in bf16 (8 lanes -> use a 32-lane exp on a padded vector)
      alignas(64) float dm32[32];
      aie::store_v(dm32, aie::zeros<float, 32>());
      aie::store_v(dm32, aie::concat(dm, aie::zeros<float, 8>()).template grow<32>());
      B32 sc32 = exp_srs(aie::accum<accfloat, 32>(aie::load_v<32>(dm32)));
      alignas(64) bfloat16 sc8[32];
      aie::store_v(sc8, sc32);
      aie::vector<float, 8> scf = aie::accum<accfloat, 8>(aie::load_v<8>(sc8)).template to_vector<float>();
      aie::store_v(g_sumv + rb * 8, aie::mul(aie::load_v<8>(g_sumv + rb * 8), scf).template to_vector<float>());
      B64 srow = rows8(sc8);
      float *orow = o_rows(rb);
      for (int j = 0; j < NB; j++) {
        A64 a(aie::load_v<64>(orow + j * 64));
        B64 hi = BF(a);
        B64 lo = BF(aie::mac(a, hi, (bfloat16)-1.0f));
        A64 pr = aie::mul(hi, srow);
        pr = aie::mac(pr, lo, srow);
        aie::store_v(orow + j * 64, pr.template to_vector<float>());
      }
    }
    // e = bf(exp(bf(s - m))) masked; row sums; PV
    B64 mrow = rows8(g_maxb + rb * 8);
    aie::vector<float, 8> rs = aie::load_v<8>(g_sumv + rb * 8);
    for (int cb = 0; cb < CB; cb++) {
      B64 d = aie::sub(aie::load_v<64>(s_rb + cb * 64), mrow);
#ifdef PROF_NOEXP
      B32 e0 = d.template extract<32>(0), e1 = d.template extract<32>(1);  // timing only
#else
      B32 e0 = exp_srs(aie::accum<accfloat, 32>(d.template extract<32>(0)));
      B32 e1 = exp_srs(aie::accum<accfloat, 32>(d.template extract<32>(1)));
#endif
      B64 e = BF(aie::mul(aie::concat(e0, e1), aie::load_v<64>(maskv + cb * mstride)));
      aie::store_v(e_rb + cb * 64, e);
      rs = aie::add(rs, rowsum8(e));
    }
    aie::store_v(g_sumv + rb * 8, rs);
    float *orow = o_rows(rb);
    for (int j = 0; j < NB; j++) {
      MM acc((A64(aie::load_v<64>(orow + j * 64))));
#ifndef PROF_NOPV
#pragma unroll
      for (int cb = 0; cb < CB; cb++) acc.mac(aie::load_v<64>(e_rb + cb * 64), aie::load_v<64>(vc + (cb * NB + j) * 64));
#endif
      aie::store_v(orow + j * 64, acc.template to_vector<float>());
    }
  }
  if (++g_chunk == NCH) g_chunk = 0;
  event1();
}

void attn_copyq(bfloat16 *q, bfloat16 *a) {  // the Q object -> resident Q tile
  for (int i = 0; i < QT * HD / 32; i++) aie::store_v(q + 32 * i, aie::load_v<32>(a + 32 * i));
}

void attn_end(bfloat16 *o) {  // o = bf(O * bf(1/sum)); rows 32..63 from o_hi, rows 0..31 in place (fp32 at 4i -> bf16 at 2i)
  alignas(64) bfloat16 rinv[QT];
  for (int r = 0; r < QT; r++) rinv[r] = (bfloat16)(1.0f / g_sumv[r]);
  for (int rb = 0; rb < RB; rb++) {
    B64 rrow = rows8(rinv + rb * 8);
    const float *src = o_rows(rb);
    for (int j = 0; j < NB; j++) {
      A64 a(aie::load_v<64>(src + j * 64));
      B64 hi = BF(a);
      B64 lo = BF(aie::mac(a, hi, (bfloat16)-1.0f));
      A64 pr = aie::mul(hi, rrow);
      pr = aie::mac(pr, lo, rrow);
      aie::store_v(o + (rb * NB + j) * 64, BF(pr));  // in-place for rb < 4: the bf16 block (128 B) at 2x lands before/at its fp32 source
    }
  }
}
}
