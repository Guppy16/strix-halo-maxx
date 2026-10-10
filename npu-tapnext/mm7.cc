// design7 GEMM microkernel: mm_f32acc.cc's BLK12 path only, with the accumulator split generalised to any m
// (stripes of 16 rows must not straddle the split) and a bias epilogue.
//
// Per core the C tile is (DIM_M, DIM_N) in 8x8 blocks, block-row major [m/8][n/8][8][8]. The fp32 accumulator
// lives in the bf16 output buffer for rows [0, ACC_LO) (m*n*2 bytes hold m/2 rows of fp32) and in acc_hi for
// rows [ACC_LO, m). acc_store_bias adds the N-tile's bias (host-tiled: 8 vectors of 64 bf16, one per 8-column
// block) in the MAC unit, rounds to bf16 (RNE) and compacts the tile in place.
//
// A sub-tile: [2][k/8][8 rows x 1 bfp block] (row-half major, pack.py half=True); B tile: [n/16][k/8][2][8 cols].
#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

template <typename T, unsigned N, aie_dm_resource R>
using InBufStream = aie::block_vector_restrict_input_buffer_stream<T, N, R>;

constexpr int m = DIM_M;
constexpr int n = DIM_N;
constexpr int k = DIM_K;
constexpr int rho = RHO;
constexpr int m_a = m / rho;
constexpr int r = 8, s = 8, t = 8;
constexpr int ACC_LO = ((m / 2) / 16) * 16;  // rows kept in the output buffer (fp32 needs 2 bf16 slots per row)
static_assert(m_a == 16 && n % 16 == 0 && k % 32 == 0);
static_assert(ACC_LO * n * 4 <= m * n * 2, "the low half must fit the bf16 buffer");

alignas(aie::vector_decl_align) static float acc_hi[(m - ACC_LO) * n];
static int g_stripe = 0;

static inline float *acc_rows(bfloat16 *c_out, int row) {
  return row < ACC_LO ? (float *)c_out + row * n : acc_hi + (row - ACC_LO) * n;
}

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
  // 1x2 register block (8 rows x 16 cols): 2 live accumulators + 2 prefetched for the next block.
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

// fp32 accumulator + bias -> bf16 tile (RNE), same blocked layout, compacted in place.
// bias: 8 x 64 bf16, vector cb = the bias of columns [8 cb, 8 cb + 8) repeated for the block's 8 rows.
void acc_store_bias(bfloat16 *c_out, bfloat16 *bias) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  using B32 = aie::vector<bfloat16, 32>;
  const B32 ones = aie::broadcast<bfloat16, 32>(1.0f);
  auto emit = [&](const float *src, bfloat16 *dst, int cb) __attribute__((always_inline)) {
    // one 8x8 block = 64 fp32 -> two 32-lane bf16 MACs with the bias (exact in fp32) -> bf16
    aie::accum<accfloat, 32> a0(aie::load_v<32>(src)), a1(aie::load_v<32>(src + 32));
    a0 = aie::mac(a0, aie::load_v<32>(bias + 64 * cb), ones);
    a1 = aie::mac(a1, aie::load_v<32>(bias + 64 * cb + 32), ones);
    aie::store_v(dst, a0.template to_vector<bfloat16>());
    aie::store_v(dst + 32, a1.template to_vector<bfloat16>());
  };
  float *src = (float *)c_out;
  for (int i = 0; i < ACC_LO * n / 64; i++)  // in place: the 64-wide fp32 load at byte 256 i precedes the bf16 store at 128 i
    emit(src + 64 * i, c_out + 64 * i, i % (n / 8));
  for (int i = 0; i < (m - ACC_LO) * n / 64; i++)
    emit(acc_hi + 64 * i, c_out + ACC_LO * n + 64 * i, i % (n / 8));
}
}
