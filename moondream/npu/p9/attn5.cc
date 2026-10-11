// Fused attention v5 = v4b (bfp16ebs8 operands, one pass, online softmax, row sums from a ones panel) with the
// exp rewritten as a software pipeline: one row block's 384 exponentials go through 3 stage loops of 12 vectors
// with the intermediates staged in L1, so each stage runs at the unit's throughput instead of the dependent
// chain's latency (v4b: ~146 cycles per 32-lane exp, 10.5k cycles of the 21k per chunk).
// Other v5 changes: the chunk max is not masked (padded keys score 0, which only shifts the stabiliser; the
// softmax is shift-invariant as long as P and the sum use the same m, which they do), so the masks only zero P in
// the last chunk; the "minus infinity" sentinel is -64 (scores are O(1)), which keeps every exp argument inside
// the range the integer exponent path handles with no clamps.
// Layouts: q [6 rb] groups of 9 panels (704 B each); kt [6 cb] groups (K natural [keys][dims], op_transpose in
// the mmul); vc [9 j] groups of 6 V^T panels [8 dims][8 keys] (448 B each); o (bf16) [6 rb][9 j][64].
#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

constexpr int QT = 48, HD = 72, KC = 48;
constexpr int NB = HD / 8, RB = QT / 8, CB = KC / 8;
constexpr int NKEYS = ATTN_KEYS;
constexpr int NCH = (NKEYS + KC - 1) / KC;  // 16
constexpr float NEG = -64.0f;               // "minus infinity" for masked keys and the initial running max

using MM = aie::mmul<8, 8, 8, bfp16ebs8>;
using BP = aie::block_vector<bfp16ebs8, 64>;
template <aie_dm_resource R>
using InS = aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, 64, R>;
using OutS = aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::none>;
constexpr int PB = 72;    // bytes per bfp16ebs8 8x8 panel
constexpr int GQ = 704;   // Q / K group: 9 panels (648 B) padded to 704 B (the streams need 64 B-aligned seeks)
constexpr int GV = 448;   // V^T group: 6 panels (432 B) padded to 448 B
constexpr int OBJB = RB * GQ;  // 4224 B per Q / K / V object
using B32 = aie::vector<bfloat16, 32>;
using B64 = aie::vector<bfloat16, 64>;
using I32 = aie::vector<int32_t, 32>;
using A32 = aie::accum<accfloat, 32>;
using A64 = aie::accum<accfloat, 64>;
#define BF(e) ((e).template to_vector<bfloat16>())
constexpr int NV = CB * 64 / 32;  // 12 vectors of 32 per row block

alignas(64) static float o_hi[(QT / 2) * HD];   // fp32 O rows 24..47; rows 0..23 live in the bf16 output object
alignas(64) static float o_sum[RB * 64];         // fp32 row sums of the bfp-rounded P (10th O block, column 0 of each row)
alignas(64) static bfp16ebs8 ones_grp[CB * PB];  // V^T panel group whose dim 0 is all ones
alignas(64) static bfloat16 g_maxb[QT];          // running row max, bf16
alignas(64) static float g_maxv[QT];             // running row max as 8-lane fp32 vectors
alignas(64) static bfloat16 mask_full[64], mask_last[CB * 64];  // P masks: full chunk = one row of ones (stride 0)
alignas(64) static bfloat16 s_rb[CB * 64];       // scores of one row block, [cb][64]
alignas(64) static bfp16ebs8 e_rb[CB * PB];      // P of one row block as bfp16 panels
alignas(64) static int32_t qz_buf[NV * 32];      // exp stage 1 -> 2: round(256 * z)
alignas(64) static int32_t n_buf[NV * 32];       // exp stage 2 -> 3: integer part
alignas(64) static bfloat16 f_buf[NV * 32];      // exp stage 2 -> 3: fraction in [-0.5, 0.5)
alignas(64) static bfloat16 m_buf[64];           // the row block's max, one value per 8-lane row
static bfloat16 g_scale;
static int g_chunk;
static float *g_o_lo;

static inline float *o_rows(int rb) { return rb < RB / 2 ? g_o_lo + rb * NB * 64 : o_hi + (rb - RB / 2) * NB * 64; }

// 2^f for f in [-0.5, 0.5) as a bf16 Horner cubic (Taylor, error < 7e-4), exponent assembled with int ops
static inline B32 exp2_assemble(B32 f, I32 n) {
  A32 p = aie::mul(f, bfloat16(0.05550411f));
  p = aie::mac(A32(aie::broadcast<float, 32>(0.24022651f)), f, BF(p));
  p = aie::mac(A32(aie::broadcast<float, 32>(0.69314718f)), f, BF(p));
  p = aie::mac(A32(aie::broadcast<float, 32>(1.0f)), f, BF(p));
  I32 ebits = aie::add(aie::upshift(n, 23), p.template to_vector<float>().template cast_to<int32_t>());
  return A32(ebits.template cast_to<float>()).template to_vector<bfloat16>();
}

// e = bf(exp(bf(s - m))) for one row block, staged: s_rb (bf16) -> P panels e_rb (bfp16, masked)
static inline void exp_rb(B64 mrow, const bfloat16 *maskv, int mstride) {
  const bfloat16 L_HI = 1.4426950408889634f;
  const bfloat16 L_LO = 1.4426950408889634f - (float)L_HI;
  aie::store_v(m_buf, mrow);
#pragma unroll
  for (int v = 0; v < NV; v++) {  // stage 1: z = d log2e (bf16 hi/lo products), q = round(256 z)
    B32 d = aie::sub(aie::load_v<32>(s_rb + v * 32), aie::load_v<32>(m_buf + (v & 1) * 32));
    A32 za = aie::mul(d, L_HI);
    za = aie::mac(za, d, L_LO);
    aie::store_v(qz_buf + v * 32, aie::to_fixed<int32_t>(za.template to_vector<float>(), 8));
  }
#pragma unroll
  for (int v = 0; v < NV; v++) {  // stage 2: n = round(z), f = z - n in [-0.5, 0.5) exact in bf16
    I32 qz = aie::load_v<32>(qz_buf + v * 32);
    I32 n = aie::downshift(aie::add(qz, aie::broadcast<int32_t, 32>(128)), 8);
    I32 fr = aie::sub(qz, aie::upshift(n, 8));
    aie::store_v(n_buf + v * 32, n);
    aie::store_v(f_buf + v * 32, A32(aie::to_float(fr, 8)).template to_vector<bfloat16>());
  }
  OutS es(e_rb);
#pragma unroll
  for (int cb = 0; cb < CB; cb++) {  // stage 3: 2^f * 2^n, mask, bfp panel
    B32 e0 = exp2_assemble(aie::load_v<32>(f_buf + cb * 64), aie::load_v<32>(n_buf + cb * 64));
    B32 e1 = exp2_assemble(aie::load_v<32>(f_buf + cb * 64 + 32), aie::load_v<32>(n_buf + cb * 64 + 32));
    B64 e = BF(aie::mul(aie::concat(e0, e1), aie::load_v<64>(maskv + cb * mstride)));
    es.push(A64(e).template to_vector<bfp16ebs8>());
  }
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

// rescale O and the sums of one row block by bf(exp(dm)), dm = m_old - m_new in [-64, 0] (8 rows)
static __attribute__((noinline)) void rescale_rb(int rb, aie::vector<float, 8> dm) {
  alignas(64) bfloat16 d32[32];
  aie::store_v(d32, aie::zeros<bfloat16, 32>());
  aie::store_v(d32, aie::accum<accfloat, 8>(dm).template to_vector<bfloat16>().template grow<32>());
  B32 d = aie::load_v<32>(d32);  // bf16 dm: the factor's error is common to O and the sum, so it cancels
  const bfloat16 L_HI = 1.4426950408889634f;
  const bfloat16 L_LO = 1.4426950408889634f - (float)L_HI;
  A32 za = aie::mul(d, L_HI);
  za = aie::mac(za, d, L_LO);
  I32 qz = aie::to_fixed<int32_t>(za.template to_vector<float>(), 8);
  I32 n = aie::downshift(aie::add(qz, aie::broadcast<int32_t, 32>(128)), 8);
  B32 f = A32(aie::to_float(aie::sub(qz, aie::upshift(n, 8)), 8)).template to_vector<bfloat16>();
  alignas(64) bfloat16 sc8[32];
  aie::store_v(sc8, exp2_assemble(f, n));
  B64 srow = rows8(sc8);
  float *orow = o_rows(rb);
  for (int j = 0; j <= NB; j++) {  // 9 O blocks + the sum block, bf16 hi/lo products in the MAC unit
    float *oj = j < NB ? orow + j * 64 : o_sum + rb * 64;
    A64 a(aie::load_v<64>(oj));
    B64 hi = BF(a);
    B64 lo = BF(aie::mac(a, hi, (bfloat16)-1.0f));
    A64 pr = aie::mul(hi, srow);
    pr = aie::mac(pr, lo, srow);
    aie::store_v(oj, pr.template to_vector<float>());
  }
}

extern "C" {

void attn_begin(bfp16ebs8 *q, bfloat16 *o) {  // o: the output object, hosts fp32 O rows 0..23 during the item
  g_o_lo = (float *)o;
  for (int r = 0; r < QT; r++) { g_maxb[r] = (bfloat16)NEG; g_maxv[r] = NEG; }
  for (int i = 0; i < (QT / 2) * HD / 32; i++) { aie::store_v(g_o_lo + 32 * i, aie::zeros<float, 32>()); aie::store_v(o_hi + 32 * i, aie::zeros<float, 32>()); }
  for (int i = 0; i < RB * 64 / 32; i++) aie::store_v(o_sum + 32 * i, aie::zeros<float, 32>());
  { uint8_t *p = (uint8_t *)ones_grp;  // panel rows [exponent][8 mantissas], value = m * 2^(E - 133): 1.0 = 64 * 2^(127 - 133)
    for (int i = 0; i < CB * PB; i++) p[i] = 0;
    for (int cb = 0; cb < CB; cb++) { p[cb * PB] = 127; for (int l = 1; l < 9; l++) p[cb * PB + l] = 64; } }
  const int valid_last = NKEYS - (NCH - 1) * KC;
  for (int cb = 0; cb < CB; cb++)
    for (int i = 0; i < 8; i++)
      for (int l = 0; l < 8; l++) {
        const int idx = cb * 64 + i * 8 + l;
        if (cb == 0) mask_full[idx] = (bfloat16)1.0f;
        mask_last[idx] = cb * 8 + l < valid_last ? (bfloat16)1.0f : (bfloat16)0.0f;
      }
  g_scale = (bfloat16)ATTN_SCALE;
  g_chunk = 0;
}

// one 48-key chunk: per row block: scores -> chunk max -> rescale if the running max grew -> P -> P [V | 1]
void attn_chunk(bfp16ebs8 *q, bfp16ebs8 *kt, bfp16ebs8 *vc) {
  event0();
  const bool last = g_chunk == NCH - 1;
  const bfloat16 *maskv = last ? mask_last : mask_full;
  const int mstride = last ? 64 : 0;
  for (int rb = 0; rb < RB; rb++) {
    B64 mx;
#ifndef PROF_NOSCORES
    for (int cb = 0; cb < CB; cb++) {  // s_rb[cb] = bf(bf(Q[rb] K[cb]^T) * scale)
      MM acc;
      InS<aie_dm_resource::a> qs(q);
      qs.seek_byte(rb * GQ);
      InS<aie_dm_resource::b> ks(kt);
      ks.seek_byte(cb * GQ);
#pragma unroll
      for (int j = 0; j < NB; j++) {
        BP a = qs.pop();
        BP bt = ks.pop();
        if (j == 0) acc.mul(a, aie::op_transpose(bt)); else acc.mac(a, aie::op_transpose(bt));
      }
      B64 s = BF(aie::mul(BF(acc), g_scale));
      aie::store_v(s_rb + cb * 64, s);
      mx = cb == 0 ? s : aie::max(mx, s);
    }
#else
    mx = aie::load_v<64>(s_rb);
#endif
    aie::vector<float, 8> m_old = aie::load_v<8>(g_maxv + rb * 8);
    aie::vector<float, 8> m_new = aie::max(rowmax8(mx), m_old);
    aie::vector<float, 8> dm = aie::sub(m_old, m_new);  // <= 0
    if (aie::reduce_min(dm) < 0.0f) {
      aie::store_v(g_maxv + rb * 8, m_new);
      aie::store_v(g_maxb + rb * 8, aie::accum<accfloat, 8>(m_new).template to_vector<bfloat16>());
      rescale_rb(rb, dm);
    }
#ifndef PROF_NOEXP
    exp_rb(rows8(g_maxb + rb * 8), maskv, mstride);
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
      for (int cb = 0; cb < CB; cb++) acc.mac(ps.pop(), aie::op_transpose(vs.pop()));
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

void attn_end(bfloat16 *o) {  // o = bf(O * bf(1/sum)); rows 24..47 from o_hi, rows 0..23 in place (fp32 at 4i -> bf16 at 2i)
  alignas(64) bfloat16 rinv[QT];
  for (int r = 0; r < QT; r++) rinv[r] = (bfloat16)(1.0f / o_sum[(r / 8) * 64 + (r % 8) * 8]);
  for (int rb = 0; rb < RB; rb++) {
    B64 rrow = rows8(rinv + rb * 8);
    const float *src = o_rows(rb);
    for (int j = 0; j < NB; j++) {
      A64 a(aie::load_v<64>(src + j * 64));
      B64 hi = BF(a);
      B64 lo = BF(aie::mac(a, hi, (bfloat16)-1.0f));
      A64 pr = aie::mul(hi, rrow);
      pr = aie::mac(pr, lo, rrow);
      aie::store_v(o + (rb * NB + j) * 64, BF(pr));
    }
  }
}
}
