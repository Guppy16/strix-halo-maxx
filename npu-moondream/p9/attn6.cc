// Fused attention v6 = v5 with the three hot loops rewritten for the machine:
//  * exp: no integer conversions. z = d log2e is rounded to 1/256 and to an integer by adding fp32 magic constants
//    in the MAC unit (bf16 1.0 x 49152 and 1.5*2^23); the integer part's low 16 bits are the two's-complement n,
//    so the bf16 bit pattern of 2^n is (n << 7) + 0x3F80 in int16 lanes (one 512-bit register); the fraction
//    f = round(z) - n is exact in bf16; 2^f by the same bf16 Horner cubic; e = bf(2^f) * 2^n (exact).
//    v5 spent ~98 bundles per 32 lanes, mostly in 32-lane int32 (two-register) ops, srs/ups moves and mode changes.
//  * scores: 3 key blocks share each Q pop (three independent accumulator chains instead of one 9-deep chain).
//  * PV: 2 dim blocks share each P pop (pairs; the 10th block is the ones panel that produces the row sums).
// Numerics are v5's: bfp16ebs8 operands, fp32 accumulation, one pass with an online softmax, -64 sentinel.
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

alignas(64) static float o_hi[(QT / 2) * HD];   // fp32 O rows 24..47; rows 0..23 live in the bf16 output object
alignas(64) static float o_sum[RB * 64];         // fp32 row sums of the bfp-rounded P (10th O block, column 0 of each row)
alignas(64) static bfp16ebs8 ones_grp[CB * PB];  // V^T panel group whose dim 0 is all ones
alignas(64) static bfloat16 g_maxb[QT];          // running row max, bf16
alignas(64) static float g_maxv[QT];             // running row max as 8-lane fp32 vectors
alignas(64) static bfloat16 mask_full[64], mask_last[CB * 64];  // P masks: full chunk = one row of ones (stride 0)
alignas(64) static bfloat16 s_rb[CB * 64];       // scores of one row block, [cb][64]
alignas(64) static bfp16ebs8 e_rb[CB * PB];      // P of one row block as bfp16 panels
alignas(64) static bfloat16 m_buf[64];           // the row block's max, one value per 8-lane row
static bfloat16 g_scale;
static int g_chunk;
static float *g_o_lo;

static inline float *o_rows(int rb) { return rb < RB / 2 ? g_o_lo + rb * NB * 64 : o_hi + (rb - RB / 2) * NB * 64; }

using I16 = aie::vector<int16_t, 32>;
struct ExpC {  // loop-invariant constants of the exp (built once per call; accumulators are register/stack values)
  A32 c2, c1, c0;
  B32 ones;
  ExpC() : c2(aie::broadcast<float, 32>(0.24022651f)), c1(aie::broadcast<float, 32>(0.69314718f)),
           c0(aie::broadcast<float, 32>(1.0f)), ones(aie::broadcast<bfloat16, 32>(1.0f)) {}
};
constexpr float M8 = 49152.0f;    // 1.5 * 2^15: fp32 ulp 2^-8 -> adding it rounds z to 1/256
constexpr float M0 = 12582912.0f; // 1.5 * 2^23: ulp 1 -> adding it rounds z to an integer, n in the low mantissa bits

// e = bf(exp(d)) for 32 bf16 lanes, d in [-64, 0]; 2^f Taylor cubic on f in [-0.5, 0.5] (error < 7e-4)
#ifdef DEBUG_EXP
alignas(64) static bfloat16 dbg_n[32], dbg_f[32], dbg_zr[32];
#endif
static inline B32 exp32(B32 d, const ExpC &k) {
  const bfloat16 L_HI = 1.4426950408889634f;
  const bfloat16 L_LO = 1.4426950408889634f - (float)L_HI;
  A32 za = aie::mul(d, L_HI);
  za = aie::mac(za, d, L_LO);                       // z = d log2e (fp32)
  A32 zq = aie::mac(za, k.ones, (bfloat16)M8);      // z rounded to 1/256, offset by M8
  A32 zn = aie::mac(za, k.ones, (bfloat16)M0);      // round(z), offset by M0
  A32 zr = aie::mac(zq, k.ones, (bfloat16)(-M8));   // z rounded to 1/256 (exact)
  B32 nb = BF(aie::mac(zn, k.ones, (bfloat16)(-M0)));  // n as bf16 (exact: |n| <= 92)
  B32 f = BF(aie::mac(zr, nb, (bfloat16)-1.0f));    // f = zr - n, exact in bf16 (<= 8 significant bits)
#ifdef DEBUG_EXP
  aie::store_v(dbg_n, nb); aie::store_v(dbg_f, f); aie::store_v(dbg_zr, BF(zr));
#endif
  aie::vector<int16_t, 64> h = zn.template to_vector<float>().template cast_to<int16_t>();
  I16 n16 = aie::filter_even(h, 1);                 // low halves of the fp32 bit patterns = n (two's complement)
  B32 p2n = aie::add(aie::upshift(n16, 7), aie::broadcast<int16_t, 32>(0x3F80)).template cast_to<bfloat16>();  // 2^n
  A32 p = aie::mul(f, bfloat16(0.05550411f));
  p = aie::mac(k.c2, f, BF(p));
  p = aie::mac(k.c1, f, BF(p));
  p = aie::mac(k.c0, f, BF(p));
  return BF(aie::mul(BF(p), p2n));
}

// e = bf(exp(bf(s - m))) for one row block: s_rb (bf16) -> P panels e_rb (bfp16, masked)
static inline void exp_rb(B64 mrow, const bfloat16 *maskv, int mstride, const ExpC &k) {
  aie::store_v(m_buf, mrow);
  OutS es(e_rb);
#pragma unroll
  for (int cb = 0; cb < CB; cb++) {
    B32 e0 = exp32(aie::sub(aie::load_v<32>(s_rb + cb * 64), aie::load_v<32>(m_buf)), k);
    B32 e1 = exp32(aie::sub(aie::load_v<32>(s_rb + cb * 64 + 32), aie::load_v<32>(m_buf + 32)), k);
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
  ExpC k;
  B32 d = aie::load_v<32>(d32);  // bf16 dm: the factor's error is common to O and the sum, so it cancels
  alignas(64) bfloat16 sc8[32];
  aie::store_v(sc8, exp32(d, k));
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
  const ExpC ek;
  for (int rb = 0; rb < RB; rb++) {
    B64 mx;
#ifndef PROF_NOSCORES
    for (int cb = 0; cb < CB; cb += 3) {  // s_rb[cb] = bf(bf(Q[rb] K[cb]^T) * scale); 3 key blocks per Q pop
      MM acc0, acc1, acc2;
      InS<aie_dm_resource::a> qs(q);
      qs.seek_byte(rb * GQ);
      InS<aie_dm_resource::b> ks0(kt), ks1(kt), ks2(kt);
      ks0.seek_byte(cb * GQ);
      ks1.seek_byte((cb + 1) * GQ);
      ks2.seek_byte((cb + 2) * GQ);
#pragma unroll
      for (int j = 0; j < NB; j++) {
        BP a = qs.pop();
        BP b0 = ks0.pop(), b1 = ks1.pop(), b2 = ks2.pop();
        if (j == 0) { acc0.mul(a, aie::op_transpose(b0)); acc1.mul(a, aie::op_transpose(b1)); acc2.mul(a, aie::op_transpose(b2)); }
        else { acc0.mac(a, aie::op_transpose(b0)); acc1.mac(a, aie::op_transpose(b1)); acc2.mac(a, aie::op_transpose(b2)); }
      }
      B64 s0 = BF(aie::mul(BF(acc0), g_scale)), s1 = BF(aie::mul(BF(acc1), g_scale)), s2 = BF(aie::mul(BF(acc2), g_scale));
      aie::store_v(s_rb + cb * 64, s0);
      aie::store_v(s_rb + (cb + 1) * 64, s1);
      aie::store_v(s_rb + (cb + 2) * 64, s2);
      B64 m3 = aie::max(aie::max(s0, s1), s2);
      mx = cb == 0 ? m3 : aie::max(mx, m3);
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
    exp_rb(rows8(g_maxb + rb * 8), maskv, mstride, ek);
#endif
    float *orow = o_rows(rb);
    for (int j = 0; j <= NB; j += 2) {  // 9 V^T dim blocks + the ones block (row sums of the P the mmul sees), in pairs
      float *o0 = orow + j * 64;
      float *o1 = j + 1 < NB ? orow + (j + 1) * 64 : o_sum + rb * 64;
      MM acc0((A64(aie::load_v<64>(o0)))), acc1((A64(aie::load_v<64>(o1))));
#ifndef PROF_NOPV
      InS<aie_dm_resource::a> ps(e_rb);
      InS<aie_dm_resource::b> vs0(vc), vs1(j + 1 < NB ? vc : ones_grp);
      vs0.seek_byte(j * GV);
      vs1.seek_byte(j + 1 < NB ? (j + 1) * GV : 0);
#pragma unroll
      for (int cb = 0; cb < CB; cb++) {
        BP pb = ps.pop();
        acc0.mac(pb, aie::op_transpose(vs0.pop()));
        acc1.mac(pb, aie::op_transpose(vs1.pop()));
      }
#endif
      aie::store_v(o0, acc0.template to_vector<float>());
      aie::store_v(o1, acc1.template to_vector<float>());
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
#ifdef DEBUG_EXP
  {  // o[0..1536) = exp32 of a ramp d = -i/64 (i = 0..1535): the host checks the integer/fraction split
    ExpC k;
    for (int v = 0; v < 48; v++) {
      alignas(64) bfloat16 d32[32];
      for (int l = 0; l < 32; l++) d32[l] = (bfloat16)(-(float)(v * 32 + l) / 64.0f);
      aie::store_v(o + v * 32, exp32(aie::load_v<32>(d32), k));
      if (v < 16) { aie::store_v(o + 1536 + v * 32, aie::load_v<32>(dbg_n)); aie::store_v(o + 2048 + v * 32, aie::load_v<32>(dbg_f)); aie::store_v(o + 2560 + v * 32, aie::load_v<32>(dbg_zr)); }
    }
    return;
  }
#endif
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
