// design8 GEMM microkernel (mm7's loop) + flag-driven epilogues. Per core the C tile is (DIM_M, DIM_N) in 8x8
// blocks [m/8][n/8][8][8]; the fp32 accumulator lives in the bf16 output object for rows [0, ACC_LO) and in acc_hi
// for the rest. The bias object appended to each N-tile's B stream holds 8 x 64 bf16 bias vectors and, at byte
// 1024, the epilogue flag:
//   0 plain      bf16(acc + bias) in place (Q16 k-step pieces: 18 x 1024 B)
//   1 gelu       bf16(gelu_tanh(bf16(acc + bias))) in place
//   2 gated_a    tile = [32 gate | 32 up] columns: 4 product panels per row block into the stash (first of a pair)
//   3 gated_b    the pair's k-step chunk [rb 18][own 4 | stash 4 panels] = 10,368 B at the object's start
//                (so the host packs each pair as (natural tile 2p+1, natural tile 2p): pack8.pack_gemm)
//   4 gelu_pan   gelu -> 8 panels per row block = a k-step chunk at the object's start
//   5/6 q/k_obj  3 attention objects (48 rows each: [rb 6][8 dim panels]) at 0 / 4608 / 9216
//   7 v_obj      3 V^T objects: 8 groups (dim block j) of 6 transposed panels, 448 B per group
#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"
#include "nk8.h"

template <typename T, unsigned N, aie_dm_resource R>
using InBufStream = aie::block_vector_restrict_input_buffer_stream<T, N, R>;

constexpr int m = DIM_M;
constexpr int n = DIM_N;
constexpr int k = DIM_K;
constexpr int rho = RHO;
constexpr int m_a = m / rho;
constexpr int r = 8, s = 8, t = 8;
constexpr int ACC_LO = 48;  // rows whose fp32 accumulator lives in the C object (12,288 B); the tail hosts the stash
constexpr int PB = 72;
static_assert(m_a == 16 && n % 16 == 0 && k % 32 == 0 && ACC_LO % 8 == 0);
static_assert(ACC_LO * n * 4 <= m * n * 2, "the low half must fit the bf16 buffer");

using B32 = aie::vector<bfloat16, 32>;
using B64 = aie::vector<bfloat16, 64>;
using A32 = aie::accum<accfloat, 32>;
using A64 = aie::accum<accfloat, 64>;
using OutS = aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::none>;
#define BF(e) ((e).template to_vector<bfloat16>())

alignas(aie::vector_decl_align) static float acc_hi[(m - ACC_LO) * n];
constexpr int STASH_STRIDE = 320;  // 4 panels (288 B) per row block, 64 B-aligned slots
constexpr int STASH_OFF = ACC_LO * n * 4;  // the gated stash lives in the C object behind the fp32 rows (5760 B)
constexpr int VTMP_OFF = m * n * 2 - 3584;  // the V^T object temp at the C object's end
static int g_stripe = 0;

static inline float *acc_rows(bfloat16 *c_out, int row) {
  return row < ACC_LO ? (float *)c_out + row * n : acc_hi + (row - ACC_LO) * n;
}

using nk8::ExpC;

extern "C" {

void acc_zero(bfloat16 *c_out) {
  const aie::vector<float, 32> z = aie::zeros<float, 32>();
  float *p0 = (float *)c_out;
  for (int i = 0; i < ACC_LO * n / 32; i++) aie::store_v(p0 + 32 * i, z);
  for (int i = 0; i < (m - ACC_LO) * n / 32; i++) aie::store_v(acc_hi + 32 * i, z);
  g_stripe = 0;
}

void mm_acc(bfp16ebs8 *__restrict pA_in, bfp16ebs8 *__restrict pB_in, bfloat16 *c_out) {
  event0();
  using AStream = InBufStream<bfp16ebs8, r * s, aie_dm_resource::a>;
  using BStream = InBufStream<bfp16ebs8, s * t, aie_dm_resource::b>;
  using CPtr = float __aie_dm_resource_c *__restrict;
  using MMUL = aie::mmul<r, s, t, bfp16ebs8>;
  const int stripe = g_stripe;
  g_stripe = (g_stripe == rho - 1) ? 0 : g_stripe + 1;
  for (int rb = 0; rb < m_a / r; rb++) {
    BStream b_stream(pB_in);
    CPtr pC = (CPtr)acc_rows(c_out, stripe * m_a + rb * r);
    aie::vector<float, r * t> c0 = aie::load_v<r * t>(pC);
    aie::vector<float, r * t> c1 = aie::load_v<r * t>(pC + r * t);
    AIE_PREPARE_FOR_POSTPIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(n / (2 * t))
#pragma unroll BLKU
    for (int blk = 0; blk < n / (2 * t); blk++) {
      CPtr pCb = pC + blk * (2 * r * t);
      CPtr pNext = (blk < n / (2 * t) - 1) ? pCb + 2 * r * t : pCb;
      MMUL acc0((aie::accum<accfloat, r * t>(c0)));
      MMUL acc1((aie::accum<accfloat, r * t>(c1)));
      c0 = aie::load_v<r * t>(pNext);
      c1 = aie::load_v<r * t>(pNext + r * t);
      AStream a_stream(pA_in);
      a_stream.seek(rb * (k / s));
#pragma unroll
      for (int q = 0; q < k / s; q++) {
        aie::block_vector<bfp16ebs8, r * s> A0 = a_stream.pop();
        aie::block_vector<bfp16ebs8, s * t> B0 = b_stream.pop();
        aie::block_vector<bfp16ebs8, s * t> B1 = b_stream.pop();
        acc0.mac(A0, aie::op_transpose(B0));
        acc1.mac(A0, aie::op_transpose(B1));
      }
      aie::store_v(pCb, acc0.template to_vector<float>());
      aie::store_v(pCb + r * t, acc1.template to_vector<float>());
    }
  }
  event1();
}

// one 8x8 block (64 fp32 at src) + bias(cb) -> two 32-lane bf16 vectors
static inline void blk_bias(const float *src, const bfloat16 *bias, int cb, B32 &y0, B32 &y1) {
  const B32 ones = aie::broadcast<bfloat16, 32>(1.0f);
  A32 a0(aie::load_v<32>(src)), a1(aie::load_v<32>(src + 32));
  a0 = aie::mac(a0, aie::load_v<32>(bias + 64 * cb), ones);
  a1 = aie::mac(a1, aie::load_v<32>(bias + 64 * cb + 32), ones);
  y0 = BF(a0);
  y1 = BF(a1);
}

static inline const float *blk_src(bfloat16 *c_out, int rb, int cb) { return acc_rows(c_out, rb * 8) + cb * 64; }

// fp32 accumulator + bias -> the epilogue selected by the object's flag. Returns nothing; `c_out` is the C object.
__attribute__((minsize)) void acc_epilogue(bfloat16 *c_out, bfloat16 *bias) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int flag = ((const int *)bias)[256];
  const ExpC kc;
  B32 y0, y1;
  if (flag <= 1) {  // plain / gelu, in place: block i's 64 fp32 at byte 256 i -> 64 bf16 at byte 128 i
    for (int i = 0; i < m * n / 64; i++) {
      blk_bias(blk_src(c_out, i / 8, i % 8), bias, i % 8, y0, y1);
      if (flag == 1) nk8::gelu32x2(y0, y1, kc);
      aie::store_v(c_out + 64 * i, y0);
      aie::store_v(c_out + 64 * i + 32, y1);
    }
  } else if (flag == 2 || flag == 3) {  // gated: product panels of columns [0, 32) x [32, 64)
    // block-vector output streams write whole 64 B units: 4 panels (288 B) go to a 64 B-aligned stash slot
    // (stride 320) or straight to the object's k-step chunk; the stash is copied behind them in 16 B vectors
    for (int rb = 0; rb < m / 8; rb++) {
      bfp16ebs8 *g_stash = (bfp16ebs8 *)c_out + STASH_OFF;
      bfp16ebs8 *dst = flag == 2 ? g_stash + rb * STASH_STRIDE : (bfp16ebs8 *)(c_out + rb * (8 * PB) / 2);
      // a block-vector output stream's state does not survive a function call: compute first, then push
      B64 zv[4];
      for (int cb = 0; cb < 4; cb++) {
        B32 u0, u1;
        blk_bias(blk_src(c_out, rb, cb), bias, cb, y0, y1);
        blk_bias(blk_src(c_out, rb, cb + 4), bias, cb + 4, u0, u1);
        nk8::gelu32x2(y0, y1, kc);
        zv[cb] = aie::concat(BF(aie::mul(y0, u0)), BF(aie::mul(y1, u1)));
      }
      {
        OutS ps(dst);
        for (int cb = 0; cb < 4; cb++) ps.push(A64(zv[cb]).template to_vector<bfp16ebs8>());
      }
      if (flag == 3) {  // [own 4 | stash 4] = 576 B at byte 576 rb (always behind the reads of row block rb: 2048 B per rb)
        const bfloat16 *s = (const bfloat16 *)(g_stash + rb * STASH_STRIDE);
        bfloat16 *d = (bfloat16 *)dst + 4 * PB / 2;
        for (int i = 0; i < 4 * PB / 16; i++) aie::store_v(d + 8 * i, aie::load_v<8>(s + 8 * i));
      }
    }
  } else if (flag == 4 || flag == 5 || flag == 6) {  // gelu -> panels (k-step chunk) / Q, K objects (3 x 3456 B at 4608 B)
    B64 *tmp = (B64 *)((uint8_t *)c_out + STASH_OFF);  // 8 x 128 B behind the fp32 rows (free during the epilogue)
    for (int rb = 0; rb < m / 8; rb++) {
      for (int cb = 0; cb < 8; cb++) {
        blk_bias(blk_src(c_out, rb, cb), bias, cb, y0, y1);
        if (flag == 4) nk8::gelu32x2(y0, y1, kc);
        tmp[cb] = aie::concat(y0, y1);
      }
      bfloat16 *dst = flag == 4 ? c_out + rb * (8 * PB) / 2 : c_out + ((rb / 6) * 4608 + (rb % 6) * 8 * PB) / 2;
      OutS ps((bfp16ebs8 *)dst);
      for (int cb = 0; cb < 8; cb++) ps.push(A64(tmp[cb]).template to_vector<bfp16ebs8>());
    }
  } else {  // V^T objects: per object (48 keys) 8 dim groups of 6 transposed panels; via a temp at the object's end
    bfp16ebs8 *g_stash = (bfp16ebs8 *)c_out + VTMP_OFF;
    for (int obj = 0; obj < 3; obj++) {
      for (int j = 0; j < 8; j++) {
        OutS ps(g_stash + j * 448);
        for (int cb = 0; cb < 6; cb++) {
          blk_bias(blk_src(c_out, obj * 6 + cb, j), bias, j, y0, y1);
          B64 tr = aie::transpose(aie::concat(y0, y1), 8, 8);
          ps.push(A64(tr).template to_vector<bfp16ebs8>());
        }
      }
      bfloat16 *dst = c_out + obj * 4608 / 2;
      for (int i = 0; i < 3584 / 64; i++) aie::store_v(dst + 32 * i, aie::load_v<32>((const bfloat16 *)g_stash + 32 * i));
    }
  }
}
}

// ---- attention on the GEMM cores (attention v15's online-softmax kernel) --------------------------------------
// Per (pass, head): Q objects ([rb 6][8 dim panels], 3456 B) copied from the A ring, K objects (same layout) and
// V^T objects (8 dim groups of 6 transposed panels, 448 B each) from the B fifo, O accumulated in fp32 over the
// 24 key chunks, output = bf16(O / sum) as bfp panels at the C object's start (tile t -> row blocks 6t..6t+5).
// All working memory is aliased onto the C object (18,432 B) and acc_hi (20,480 B), which the GEMM does not use
// during the attention op.
namespace at {
constexpr int QT = 48, HD = 64, KC = 48, NB = HD / 8, RB = QT / 8, CB = KC / 8;
constexpr int NKEYS = ATTN_KEYS;
constexpr int NCH = 24;
constexpr float NEG = -64.0f;
constexpr int GQ = 576, GV = 448, OBJQ = 3456;
static_assert(NKEYS % 16 == 0 && NKEYS <= NCH * KC);
using MM = aie::mmul<8, 8, 8, bfp16ebs8>;
using BP = aie::block_vector<bfp16ebs8, 64>;
template <aie_dm_resource R>
using InS = aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, 64, R>;
using B16 = aie::vector<bfloat16, 16>;
static float *g_o[2];          // O tiles: [rb 6][dim block 8][64 fp32]
static float *g_osum;          // row sums: [tile][rb][64 fp32]
static bfp16ebs8 *g_q[2];      // Q copies
static bfp16ebs8 *g_erb[2];    // P of the current row block per tile, bfp panels [cb][448 B]
static bfloat16 *g_srb;        // scores of one row block [cb][64]
static bfloat16 *g_maxb;       // running max [tile][rb][16]
static bfloat16 *g_mbuf;       // [64]
static bfp16ebs8 *g_ones;      // V^T group whose dim 0 is all ones (6 panels)
static int g_ntiles, g_chunk;
static B64 *g_etmp;           // 6 x 128 B scratch for exp_rb

static inline float *o_rows(int t, int rb) { return g_o[t] + rb * NB * 64; }
static inline B64 rows8(const bfloat16 *v8) {
  return aie::concat(aie::concat(aie::broadcast<bfloat16, 8>(v8[0]), aie::broadcast<bfloat16, 8>(v8[1]),
                                 aie::broadcast<bfloat16, 8>(v8[2]), aie::broadcast<bfloat16, 8>(v8[3])),
                     aie::concat(aie::broadcast<bfloat16, 8>(v8[4]), aie::broadcast<bfloat16, 8>(v8[5]),
                                 aie::broadcast<bfloat16, 8>(v8[6]), aie::broadcast<bfloat16, 8>(v8[7])));
}
static inline B16 rowmax8(B64 v) {
  B64 t = aie::transpose(v, 8, 8);
  B16 m = aie::max(t.template extract<16>(0), t.template extract<16>(1));
  m = aie::max(m, aie::max(t.template extract<16>(2), t.template extract<16>(3)));
  return aie::max(m, aie::shuffle_down_rotate(m, 8));
}

// e = bf(exp(bf(s - m))) for one row block -> P panels (bfp16) for the nkb valid key blocks (the rest zero)
static inline void exp_rb(int t, int rb, B64 mrow, int nkb, const ExpC &k) {
  aie::store_v(g_mbuf, mrow);
  B64 *ev = g_etmp;  // compute all key blocks first: the output stream's state does not survive the exp calls
  for (int cb = 0; cb < CB; cb++) {
    B32 e0 = aie::sub(aie::load_v<32>(g_srb + cb * 64), aie::load_v<32>(g_mbuf));
    B32 e1 = aie::sub(aie::load_v<32>(g_srb + cb * 64 + 32), aie::load_v<32>(g_mbuf + 32));
    nk8::exp32x2(e0, e1, k);
    B64 e = aie::concat(e0, e1);
    if (cb >= nkb) e = aie::zeros<bfloat16, 64>();
    ev[cb] = e;
  }
  OutS es(g_erb[t] + rb * GV);  // one 448 B group (6 panels) per row block
  for (int cb = 0; cb < CB; cb++) es.push(A64(ev[cb]).template to_vector<bfp16ebs8>());
}

static __attribute__((noinline, minsize)) void rescale_rb(int t, int rb, B16 dm) {
  ExpC k;
  B32 d = aie::concat(dm, aie::zeros<bfloat16, 16>());
  alignas(64) bfloat16 sc8[32];
  aie::store_v(sc8, nk8::exp32(d, k));
  B64 srow = rows8(sc8);
  float *orow = o_rows(t, rb);
  for (int j = 0; j <= NB; j++) {
    float *oj = j < NB ? orow + j * 64 : g_osum + (t * RB + rb) * 64;
    A64 a(aie::load_v<64>(oj));
    B64 hi = BF(a);
    B64 lo = BF(aie::mac(a, hi, (bfloat16)-1.0f));
    A64 pr = aie::mul(hi, srow);
    pr = aie::mac(pr, lo, srow);
    aie::store_v(oj, pr.template to_vector<float>());
  }
}

// scores -> running max / rescale -> P (bfp panels, one row block at a time; P kept per tile for the V phase)
static __attribute__((minsize)) void chunk_k(int t, const bfp16ebs8 *q, const bfp16ebs8 *kt, int nkb, const ExpC &ek) {
  const bfloat16 scale = (bfloat16)ATTN_SCALE;
  for (int rb = 0; rb < RB; rb++) {
    B64 mx;
    BP qreg[NB];
    {
      InS<aie_dm_resource::a> qs(q);
      qs.seek_byte(rb * GQ);
#pragma unroll
      for (int j = 0; j < NB; j++) qreg[j] = qs.pop();
    }
    for (int cb = 0; cb < nkb; cb += 2) {
      MM acc0, acc1, acc2, acc3;
      InS<aie_dm_resource::b> ks0(kt), ks1(kt);
      ks0.seek_byte(cb * GQ);
      ks1.seek_byte((cb + 1) * GQ);
#pragma unroll
      for (int j = 0; j < NB; j += 2) {
        BP b0 = ks0.pop(), b1 = ks1.pop();
        if (j == 0) { acc0.mul(qreg[j], aie::op_transpose(b0)); acc1.mul(qreg[j], aie::op_transpose(b1)); }
        else { acc0.mac(qreg[j], aie::op_transpose(b0)); acc1.mac(qreg[j], aie::op_transpose(b1)); }
        BP b2 = ks0.pop(), b3 = ks1.pop();
        if (j == 0) { acc2.mul(qreg[j + 1], aie::op_transpose(b2)); acc3.mul(qreg[j + 1], aie::op_transpose(b3)); }
        else { acc2.mac(qreg[j + 1], aie::op_transpose(b2)); acc3.mac(qreg[j + 1], aie::op_transpose(b3)); }
      }
      A64 t0 = aie::add(acc0.to_accum(), acc2.to_accum()), t1 = aie::add(acc1.to_accum(), acc3.to_accum());
      B64 s0 = BF(aie::mul(BF(t0), scale)), s1 = BF(aie::mul(BF(t1), scale));
      aie::store_v(g_srb + cb * 64, s0);
      aie::store_v(g_srb + (cb + 1) * 64, s1);
      B64 m2 = aie::max(s0, s1);
      mx = cb == 0 ? m2 : aie::max(mx, m2);
    }
    bfloat16 *mb = g_maxb + (t * RB + rb) * 16;
    B16 m_old = aie::load_v<16>(mb);
    B16 m_new = aie::max(rowmax8(mx), m_old);
    if (!aie::lt(m_old, m_new).empty()) {
      aie::store_v(mb, m_new);
      rescale_rb(t, rb, BF(aie::mac(aie::accum<accfloat, 16>(m_old), m_new, (bfloat16)-1.0f)));
    }
    exp_rb(t, rb, rows8(mb), nkb, ek);
    // the P panels of this row block go to the tile's P buffer slot rb (6 x 448 B per tile)
  }
}

// O += P [V | 1] for the 6 row blocks of tile t
static __attribute__((minsize)) void chunk_v(int t, const bfp16ebs8 *vc) {
  for (int rb = 0; rb < RB; rb++) {
    float *orow = o_rows(t, rb);
    BP preg[CB];
    {
      InS<aie_dm_resource::a> ps(g_erb[t] + rb * GV);
#pragma unroll
      for (int cb = 0; cb < CB; cb++) preg[cb] = ps.pop();
    }
    for (int j = 0; j < NB; j += 2) {  // dim block pairs
      float *o0 = orow + j * 64, *o1 = orow + (j + 1) * 64;
      MM acc0((A64(aie::load_v<64>(o0)))), acc1((A64(aie::load_v<64>(o1)))), acc2, acc3;
      InS<aie_dm_resource::b> vs0(vc), vs1(vc);
      vs0.seek_byte(j * GV);
      vs1.seek_byte((j + 1) * GV);
#pragma unroll
      for (int cb = 0; cb < CB; cb += 2) {
        acc0.mac(preg[cb], aie::op_transpose(vs0.pop()));
        acc1.mac(preg[cb], aie::op_transpose(vs1.pop()));
        if (cb == 0) { acc2.mul(preg[cb + 1], aie::op_transpose(vs0.pop())); acc3.mul(preg[cb + 1], aie::op_transpose(vs1.pop())); }
        else { acc2.mac(preg[cb + 1], aie::op_transpose(vs0.pop())); acc3.mac(preg[cb + 1], aie::op_transpose(vs1.pop())); }
      }
      aie::store_v(o0, aie::add(acc0.to_accum(), acc2.to_accum()).template to_vector<float>());
      aie::store_v(o1, aie::add(acc1.to_accum(), acc3.to_accum()).template to_vector<float>());
    }
    {  // the row sums: P times the ones panels (the same bfp-rounded P the mmul sees)
      float *os = g_osum + (t * RB + rb) * 64;
      MM acc0((A64(aie::load_v<64>(os)))), acc2;
      InS<aie_dm_resource::b> vs0(g_ones);
#pragma unroll
      for (int cb = 0; cb < CB; cb += 2) {
        acc0.mac(preg[cb], aie::op_transpose(vs0.pop()));
        if (cb == 0) acc2.mul(preg[cb + 1], aie::op_transpose(vs0.pop()));
        else acc2.mac(preg[cb + 1], aie::op_transpose(vs0.pop()));
      }
      aie::store_v(os, aie::add(acc0.to_accum(), acc2.to_accum()).template to_vector<float>());
    }
  }
}
}  // namespace at

static inline void q_map() {  // the acc_hi part of the map (valid before any C object is acquired)
  using namespace at;
  uint8_t *hb = (uint8_t *)acc_hi;
  g_o[1] = (float *)hb;                 // [0, 12288)
  g_q[0] = (bfp16ebs8 *)(hb + 12288);   // 3456 B
  g_q[1] = (bfp16ebs8 *)(hb + 15744);   // 3456 B
  g_erb[0] = (bfp16ebs8 *)(hb + 19200); // 2688 B
  g_erb[1] = (bfp16ebs8 *)(hb + 21888); // 2688 B (to 24576)
}

extern "C" {
// c_out: the C object of the first head of the pass (only used for its address; every head's object is at the
// same L1 address since the C sub-fifo is single-buffered)
__attribute__((minsize)) void attn_begin(bfloat16 *c_out, int32_t ntiles) {
  using namespace at;
#ifdef ATTN_STUB
  return;
#endif
  aie::set_rounding(aie::rounding_mode::conv_even);
  uint8_t *cb = (uint8_t *)c_out;
  // acc_hi (24,576 B, static: its address is known before the C object is acquired): O tile 1, the Q copies, the
  // P buffers. The C object (18,432 B): O tile 0, the row sums, scores, maxima, the ones panels, the exp scratch.
  // (round 3 gave g_srb 384 B: the 6 score blocks are 6 x 128 B = 768 B, so the running maxima aliased the scores
  // of key blocks 3-5 and every row block's softmax saw 16 garbage P entries: corr 0.91 / 0.69 instead of 1.0)
  g_o[0] = (float *)cb;                 // [0, 12288)
  g_osum = (float *)(cb + 12288);       // 3072 B (to 15360)
  g_srb = (bfloat16 *)(cb + 15360);     // 768 B = CB x 64 bf16 (to 16128)
  g_maxb = (bfloat16 *)(cb + 16128);    // 384 B (to 16512)
  g_mbuf = (bfloat16 *)(cb + 16512);    // 128 B (to 16640)
  g_ones = (bfp16ebs8 *)(cb + 16640);   // 432 B, 64 B-aligned (to 17072)
  g_etmp = (B64 *)(cb + 17088);         // 768 B (to 17856 <= 18432)
  static_assert(CB * 64 * 2 == 768 && 17088 + 768 <= m * n * 2);
  q_map();
  g_ntiles = ntiles;
  g_chunk = 0;
  for (int i = 0; i < 2 * RB * NB * 64 / 32; i++) aie::store_v(g_o[i / (RB * NB * 2)] + (i % (RB * NB * 2)) * 32, aie::zeros<float, 32>());
  for (int i = 0; i < 2 * RB * 64 / 32; i++) aie::store_v(g_osum + 32 * i, aie::zeros<float, 32>());
  for (int i = 0; i < 2 * RB * 16 / 32; i++) aie::store_v(g_maxb + 32 * i, aie::broadcast<bfloat16, 32>(NEG));
  uint8_t *p = (uint8_t *)g_ones;  // 1.0 = mantissa 64 at exponent 127
  for (int i = 0; i < CB * PB; i++) p[i] = 0;
  for (int c = 0; c < CB; c++) { p[c * PB] = 127; for (int l = 1; l < 9; l++) p[c * PB + l] = 64; }
}

// the A ring's 3 sub-tiles hold Q object (head hh of the current triple, tile t): copy it if it is this core's
__attribute__((minsize)) void attn_qcopy(bfp16ebs8 *a0, bfp16ebs8 *a1, bfp16ebs8 *a2, int32_t hh, int32_t t, int32_t pass, int32_t myrow) {
  using namespace at;
#ifdef ATTN_STUB
  return;
#endif
  if (hh != myrow) return;
  q_map();  // the Q copies arrive before this head's C object is acquired: only acc_hi may be written here
  const int first = pass == 0 ? 0 : 2, n = pass == 0 ? 2 : 1;
  if (t < first || t >= first + n) return;
  uint32_t *d = (uint32_t *)g_q[t - first];
  const uint32_t *s[3] = {(const uint32_t *)a0, (const uint32_t *)a1, (const uint32_t *)a2};
  for (int i = 0; i < 3; i++)
    for (int w = 0; w < 1152 / 64; w++) aie::store_v(d + i * 288 + 16 * w, aie::load_v<16>(s[i] + 16 * w));
}

__attribute__((minsize)) void attn_chunk_k(bfp16ebs8 *kobj) {
  using namespace at;
#if defined(ATTN_STUB) || defined(ATTN_STUB_K)
  return;
#endif
  event0();
  const int valid = NKEYS - g_chunk * KC;
  const int nkb = valid >= KC ? CB : (valid <= 0 ? 0 : valid / 8);
  if (nkb > 0) {
    const ExpC ek;
    for (int t = 0; t < g_ntiles; t++) chunk_k(t, g_q[t], kobj, nkb, ek);
  }
  event1();
}

__attribute__((minsize)) void attn_chunk_v(bfp16ebs8 *vobj) {
  using namespace at;
#if defined(ATTN_STUB) || defined(ATTN_STUB_V)
  return;
#endif
  const int valid = NKEYS - g_chunk * KC;
  if (valid > 0)
    for (int t = 0; t < g_ntiles; t++) chunk_v(t, vobj);
  if (++g_chunk == NCH) g_chunk = 0;
}

// o = bf(O * bf(1/sum)) -> bfp panels at the C object's start: tile t, row block rb -> [6 t + rb][8 dim panels]
__attribute__((minsize)) void attn_end(bfloat16 *c_out) {
  using namespace at;
#if defined(ATTN_STUB) || defined(ATTN_STUB_END)
  return;
#endif
  for (int t = 0; t < g_ntiles; t++) {
    for (int rb = 0; rb < RB; rb++) {
      alignas(64) bfloat16 blk[64], rinv[8];
      aie::store_v(blk, aie::inv(BF(A64(aie::load_v<64>(g_osum + (t * RB + rb) * 64)))));
      for (int i = 0; i < 8; i++) rinv[i] = blk[i * 8];
      B64 rrow = rows8(rinv);
      const float *src = o_rows(t, rb);
      OutS ps((bfp16ebs8 *)c_out + (t * RB + rb) * 8 * PB);
      for (int j = 0; j < NB; j++) {
        A64 a(aie::load_v<64>(src + j * 64));
        B64 hi = BF(a);
        B64 lo = BF(aie::mac(a, hi, (bfloat16)-1.0f));
        A64 pr = aie::mul(hi, rrow);
        pr = aie::mac(pr, lo, rrow);
        ps.push(pr.template to_vector<bfp16ebs8>());
      }
    }
  }
}
}
