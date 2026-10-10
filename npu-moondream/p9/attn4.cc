// Fused attention v3 for one (head, 64-query tile) item per core: ONE pass over 64-key chunks with an online
// softmax, K^T panels supplied by the producer (no transposes), half of the fp32 O accumulator living in the
// output object, and an exp with no emulated fp32 ops. bf16emu.py numerics as far as the single pass allows:
//   s = bf(bf(Q K^T) * scale) per chunk; running row max m (bf16); e = bf(exp(bf(s - m)));
//   O and the row sum (a 10th O block: P x a ones panel, so the sum is of the bfp-rounded P the mmul actually
//   multiplies - the hardware's fp32->bfp conversion truncates) accumulate in fp32; when m grows by dm they are rescaled by bf(exp(-dm)) (MAC-unit
//   hi/lo products, no vector selects: the rescale is a scalar branch per row block);
//   o = bf(O * bf(1/sum)) at the end.
// v4: Q, K^T and V arrive as bfp16ebs8 panels (72 B per 8x8 block, 8 values per shared exponent along the panel
// row), P is converted to bfp16 once per block, and the mmul is the native bfp16 one (the AIE API's bf16 mmul
// converts both operands to bfp16 on every call anyway). q [6 rb][9 j] panels; kt [6 cb][9 j] panels of [8 dims][8 keys];
// vc [9 j][6 cb] panels of V^T blocks [8 dims][8 keys] (groups padded to 64 B, see GQ/GV); o (bf16) [6 rb][9 j][64]. (The bfp mmul takes B as op_transpose
// of [n][k] blocks: K in its natural [keys][dims] layout, V transposed per block.)
#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

constexpr int QT = 48, HD = 72, KC = 48;  // 48-query tiles, 48-key chunks: Q, O and chunk objects are all 48 x 72
constexpr int NB = HD / 8, RB = QT / 8, CB = KC / 8;
constexpr int NKEYS = ATTN_KEYS;
constexpr int NCH = (NKEYS + KC - 1) / KC;  // 16

using MM = aie::mmul<8, 8, 8, bfp16ebs8>;
using BP = aie::block_vector<bfp16ebs8, 64>;
template <aie_dm_resource R>
using InS = aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, 64, R>;  // panel streams: seek(i) + pop()
using OutS = aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::none>;
constexpr int PB = 72;  // bytes per bfp16ebs8 8x8 panel
// The block-vector streams load 512-bit words and reset their position on seek, so every seek target must be
// 64 B-aligned: panels are grouped and each group is padded to a multiple of 64 B.
constexpr int GQ = 704;   // Q / K: one 8-row block = 9 panels (648 B) padded to 704 B; 6 groups = 4224 B
constexpr int GV = 448;   // V^T: one dim block = 6 panels (432 B, [cb]) padded to 448 B; 9 groups = 4032 B
constexpr int OBJB = RB * GQ;  // 4224 B per Q / K / V object (V padded to the same size)
using B32 = aie::vector<bfloat16, 32>;
using B64 = aie::vector<bfloat16, 64>;
using A32 = aie::accum<accfloat, 32>;
using A64 = aie::accum<accfloat, 64>;
#define BF(e) ((e).template to_vector<bfloat16>())

alignas(64) static float o_hi[(QT / 2) * HD];   // fp32 O rows 24..47; rows 0..23 live in the bf16 output object
alignas(64) static bfloat16 g_maxb[QT];          // running row max, bf16
alignas(64) static float g_maxv[QT];             // running row max as 8-lane fp32 vectors
alignas(64) static float o_sum[RB * 64];         // fp32 row sums of the bfp-rounded P, as a 10th O block (column 0 of each row)
alignas(64) static bfp16ebs8 ones_grp[CB * PB];  // a V^T panel group whose dim 0 is all ones: P x ones = sum of the P the mmul sees
alignas(64) static bfloat16 mask_full[64], neg_full[64], mask_last[CB * 64], neg_last[CB * 64];  // full-chunk masks: one row, stride 0
alignas(64) static bfloat16 s_rb[CB * 64];       // scores of one row block, [cb][64]
alignas(64) static bfp16ebs8 e_rb[CB * PB];      // exp of one row block as bfp16 panels
static bfloat16 g_scale;
#ifdef DEBUG_DUMP
alignas(64) static bfloat16 dbg_s[4 * CB * 64];   // chunk-0 scores of row blocks 0..3
alignas(64) static bfp16ebs8 dbg_p[RB * CB * PB]; // chunk-0 P panels of all row blocks
#endif
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
  aie::vector<int32_t, 32> n = aie::downshift(aie::add(qz, aie::broadcast<int32_t, 32>(128)), 8);  // round(z): f in [-0.5, 0.5) halves the cubic's error
  aie::vector<int32_t, 32> fr = aie::sub(qz, aie::upshift(n, 8));      // -128..127
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

extern "C" {

void attn_begin(bfp16ebs8 *q, bfloat16 *o) {  // o: the output object, hosts fp32 O rows 0..23 during the item
  g_o_lo = (float *)o;
  for (int r = 0; r < QT; r++) { g_maxb[r] = (bfloat16)(-1e30f); g_maxv[r] = -1e30f; }
  for (int i = 0; i < RB * 64 / 32; i++) aie::store_v(o_sum + 32 * i, aie::zeros<float, 32>());
  { uint8_t *p = (uint8_t *)ones_grp;  // bfp16ebs8 panel rows: [exponent][8 mantissas], value = m * 2^(E - 133); 1.0 = 64 * 2^(127 - 133)
    for (int i = 0; i < CB * PB; i++) p[i] = 0;
    for (int cb = 0; cb < CB; cb++) { p[cb * PB] = 127; for (int l = 1; l < 9; l++) p[cb * PB + l] = 64; } }
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
void attn_chunk(bfp16ebs8 *q, bfp16ebs8 *kt, bfp16ebs8 *vc) {  // kt = K blocks (natural), vc = V^T blocks
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
      InS<aie_dm_resource::a> qs(q);
      qs.seek_byte(rb * GQ);
      InS<aie_dm_resource::b> ks(kt);
      ks.seek_byte(cb * GQ);
#pragma unroll
      for (int j = 0; j < NB; j++) {
        BP a = qs.pop();
        BP bt = ks.pop();
        if (j == 0) acc.mul(a, aie::op_transpose(bt)); else acc.mac(a, aie::op_transpose(bt));  // K blocks in their natural [keys][dims] layout
      }
      B64 s = BF(aie::mul(BF(acc), g_scale));
      aie::store_v(s_rb + cb * 64, s);
      B64 sm = aie::add(BF(aie::mul(s, aie::load_v<64>(maskv + cb * mstride))), aie::load_v<64>(negv + cb * mstride));
      mx = cb == 0 ? sm : aie::max(mx, sm);
    }
#ifdef DEBUG_DUMP
    if (g_chunk == 0 && rb < 4) for (int i = 0; i < CB * 64; i++) dbg_s[rb * CB * 64 + i] = s_rb[i];
#endif
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
      B64 srow = rows8(sc8);
      float *orow = o_rows(rb);
      for (int j = 0; j <= NB; j++) {  // 9 O blocks + the sum block
        float *oj = j < NB ? orow + j * 64 : o_sum + rb * 64;
        A64 a(aie::load_v<64>(oj));
        B64 hi = BF(a);
        B64 lo = BF(aie::mac(a, hi, (bfloat16)-1.0f));
        A64 pr = aie::mul(hi, srow);
        pr = aie::mac(pr, lo, srow);
        aie::store_v(oj, pr.template to_vector<float>());
      }
    }
    // e = bf(exp(bf(s - m))) masked; row sums; PV
    B64 mrow = rows8(g_maxb + rb * 8);
    OutS es(e_rb);
    for (int cb = 0; cb < CB; cb++) {
      B64 d = aie::sub(aie::load_v<64>(s_rb + cb * 64), mrow);
#ifdef PROF_NOEXP
      B32 e0 = d.template extract<32>(0), e1 = d.template extract<32>(1);  // timing only
#else
      B32 e0 = exp_srs(aie::accum<accfloat, 32>(d.template extract<32>(0)));
      B32 e1 = exp_srs(aie::accum<accfloat, 32>(d.template extract<32>(1)));
#endif
      B64 e = BF(aie::mul(aie::concat(e0, e1), aie::load_v<64>(maskv + cb * mstride)));
      es.push(aie::accum<accfloat, 64>(e).template to_vector<bfp16ebs8>());  // P panel cb
    }
#ifdef DEBUG_DUMP
    if (g_chunk == 0) { const uint32_t *sp = (const uint32_t *)e_rb; uint32_t *dp = (uint32_t *)dbg_p + rb * CB * PB / 4; for (int i = 0; i < CB * PB / 4; i++) dp[i] = sp[i]; }
#endif
    float *orow = o_rows(rb);
    for (int j = 0; j <= NB; j++) {  // 9 V^T dim blocks + the ones block (row sums of the P the mmul sees)
      float *oj = j < NB ? orow + j * 64 : o_sum + rb * 64;
      MM acc((A64(aie::load_v<64>(oj))));
#ifndef PROF_NOPV
      InS<aie_dm_resource::a> ps(e_rb);
      InS<aie_dm_resource::b> vs(j < NB ? vc : ones_grp);
      vs.seek_byte(j < NB ? j * GV : 0);
#pragma unroll
      for (int cb = 0; cb < CB; cb++) acc.mac(ps.pop(), aie::op_transpose(vs.pop()));  // V^T blocks [dims][keys], [j][cb]
#endif
      aie::store_v(oj, acc.template to_vector<float>());
    }
  }
  if (++g_chunk == NCH) g_chunk = 0;
  event1();
}

void attn_copyq(bfp16ebs8 *q, bfp16ebs8 *a) {  // the Q object (6 groups, 4224 B) -> resident Q tile
  const uint32_t *s = (const uint32_t *)a; uint32_t *d = (uint32_t *)q;
  for (int i = 0; i < OBJB / 4; i++) d[i] = s[i];
}

void attn_end(bfloat16 *o) {  // o = bf(O * bf(1/sum)); rows 32..63 from o_hi, rows 0..31 in place (fp32 at 4i -> bf16 at 2i)
#ifdef DEBUG_DUMP
  for (int i = 0; i < 4 * CB * 64; i++) o[i] = dbg_s[i];
  { const uint16_t *sp = (const uint16_t *)dbg_p; uint16_t *dp = (uint16_t *)(o + 4 * CB * 64); for (int i = 0; i < RB * CB * PB / 2; i++) dp[i] = sp[i]; }
  return;
#endif
  alignas(64) bfloat16 rinv[QT];
  for (int r = 0; r < QT; r++) rinv[r] = (bfloat16)(1.0f / o_sum[(r / 8) * 64 + (r % 8) * 8]);  // column 0 of the sum block
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
