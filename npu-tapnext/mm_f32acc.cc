// bfp16ebs8 x bfp16ebs8 GEMM microkernel with an fp32 accumulator kept for the
// whole K reduction; C leaves the core as bf16 once, at the end.
//
// Per core the C tile is (DIM_M, DIM_N). It is fed as ATB: every k-step brings
// one B tile (DIM_K, DIM_N) and RHO A sub-tiles (DIM_M/RHO, DIM_K), one call
// per A sub-tile; calls rotate over the RHO row stripes of C.
//
// L1 is too small for an fp32 tile plus a separate bf16 output buffer, so the
// fp32 accumulator is split: rows [0, M/2) live inside the bf16 output buffer
// (which is exactly M*N*2 bytes = M/2 rows of fp32), rows [M/2, M) in a static
// staging buffer. acc_store compacts the first half in place, then writes the
// second half behind it.
//
// Layouts (all in 8x8 blocks, block-row major: [M/8][N/8][8][8]):
//   A sub-tile: [m_a/16][k/8][2][8 rows x 1 bfp block] (row blocks along K)
//   B tile:     [n/16][k/8][2][8 cols x 1 bfp block]   (B^T blocks along K)
//   C / acc:    [m/8][n/8][8][8]
// The host packs A and B in exactly the order the core pops them.

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
#ifndef PIPE
#define PIPE 0
#endif
#ifndef PIPE_II
#define PIPE_II 40
#endif

static_assert(m_a % 16 == 0 && n % 16 == 0 && k % 32 == 0);
static_assert((m / 2) % 16 == 0, "acc split must fall on a 16-row boundary");

alignas(aie::vector_decl_align) static float acc_hi[(m / 2) * n];

static int g_stripe = 0;
// Always 0. Read through volatile so the compiler cannot see that every block
// re-reads the same A band and hoist those pops into a stack spill.
static volatile int g_zero = 0;

// fp32 accumulator rows [row, row+16) as a pointer: first half in the output
// buffer, second half in acc_hi.
static inline float *acc_rows(bfloat16 *c_out, int row) {
  return row < m / 2 ? (float *)c_out + row * n : acc_hi + (row - m / 2) * n;
}

extern "C" {
#ifdef ASM_BLK
void mm_tile_asm(bfp16ebs8 *pA, bfp16ebs8 *pB, float *pC);
#endif

void acc_zero(bfloat16 *c_out) {
  const aie::vector<float, 32> z = aie::zeros<float, 32>();
  float *p0 = (float *)c_out;
  for (int i = 0; i < (m / 2) * n / 32; i++)
    aie::store_v(p0 + 32 * i, z);
  for (int i = 0; i < (m / 2) * n / 32; i++)
    aie::store_v(acc_hi + 32 * i, z);
  g_stripe = 0;
}

void mm_acc(bfp16ebs8 *__restrict pA_in, bfp16ebs8 *__restrict pB_in,
            bfloat16 *c_out) {
  event0();
#ifdef NOCOMPUTE
  return;
#endif
  using AStream = InBufStream<bfp16ebs8, r * s, aie_dm_resource::a>;
  using BStream = InBufStream<bfp16ebs8, s * t, aie_dm_resource::b>;
  using CPtr = float __aie_dm_resource_c *__restrict;
  using MMUL = aie::mmul<r, s, t, bfp16ebs8>;

#ifdef OPAQUE_A
  const int z = g_zero;
#else
  const int z = 0;
#endif
  const int stripe = g_stripe;
  g_stripe = (g_stripe == rho - 1) ? 0 : g_stripe + 1;
#ifdef ASM_BLK  // phase 11: the register-blocked loop as a hand-scheduled routine (gemm_blk.s, gen_gemm.py)
  mm_tile_asm(pA_in, pB_in, acc_rows(c_out, stripe * m_a));
  event1();
  return;
#endif

#ifdef BLK12U
  // BLK12 with two 8x16 blocks per iteration: four accumulator chains, the
  // next pair's accumulators prefetched, 32 B pops / 32 MACs per iteration.
  for (int rb = 0; rb < m_a / r; rb++) {
    BStream b_stream(pB_in);
    CPtr pC = (CPtr)acc_rows(c_out, stripe * m_a + rb * r);
    aie::vector<float, r * t> c0 = aie::load_v<r * t>(pC);
    aie::vector<float, r * t> c1 = aie::load_v<r * t>(pC + r * t);
    aie::vector<float, r * t> c2 = aie::load_v<r * t>(pC + 2 * r * t);
    aie::vector<float, r * t> c3 = aie::load_v<r * t>(pC + 3 * r * t);
    AIE_LOOP_MIN_ITERATION_COUNT(n / (4 * t))
    for (int blk = 0; blk < n / (4 * t); blk++) {
      CPtr pCb = pC + blk * (4 * r * t);
      CPtr pNext = (blk < n / (4 * t) - 1) ? pCb + 4 * r * t : pCb;
      MMUL acc0((aie::accum<accfloat, r * t>(c0)));
      MMUL acc1((aie::accum<accfloat, r * t>(c1)));
      MMUL acc2((aie::accum<accfloat, r * t>(c2)));
      MMUL acc3((aie::accum<accfloat, r * t>(c3)));
      c0 = aie::load_v<r * t>(pNext);
      c1 = aie::load_v<r * t>(pNext + r * t);
      c2 = aie::load_v<r * t>(pNext + 2 * r * t);
      c3 = aie::load_v<r * t>(pNext + 3 * r * t);
      AStream a_stream(pA_in);
      a_stream.seek(rb * (k / s) + z * blk);
#pragma unroll
      for (int q = 0; q < k / s; q++) {
        aie::block_vector<bfp16ebs8, r * s> A0 = a_stream.pop();
        aie::block_vector<bfp16ebs8, s * t> B0 = b_stream.pop();
        aie::block_vector<bfp16ebs8, s * t> B1 = b_stream.pop();
        acc0.mac(A0, aie::op_transpose(B0));
        acc1.mac(A0, aie::op_transpose(B1));
      }
      AStream a2_stream(pA_in);
      a2_stream.seek(rb * (k / s) + z * blk);
#pragma unroll
      for (int q = 0; q < k / s; q++) {
        aie::block_vector<bfp16ebs8, r * s> A0 = a2_stream.pop();
        aie::block_vector<bfp16ebs8, s * t> B0 = b_stream.pop();
        aie::block_vector<bfp16ebs8, s * t> B1 = b_stream.pop();
        acc2.mac(A0, aie::op_transpose(B0));
        acc3.mac(A0, aie::op_transpose(B1));
      }
      aie::store_v(pCb, acc0.template to_vector<float>());
      aie::store_v(pCb + r * t, acc1.template to_vector<float>());
      aie::store_v(pCb + 2 * r * t, acc2.template to_vector<float>());
      aie::store_v(pCb + 3 * r * t, acc3.template to_vector<float>());
    }
  }
  event1();
  return;
#endif
#ifdef BLK12H
  // BLK12 with only HOIST k-sub A panels loop-invariant (hoisted into ex
  // registers); the rest are popped behind an opaque seek. Frees ex registers
  // for B panels in flight (12 exist; BLK12 pinned 8 with A).
#ifndef HOIST
#define HOIST 4
#endif
  for (int rb = 0; rb < m_a / r; rb++) {
    BStream b_stream(pB_in);
    CPtr pC = (CPtr)acc_rows(c_out, stripe * m_a + rb * r);
    aie::vector<float, r * t> c0 = aie::load_v<r * t>(pC);
    aie::vector<float, r * t> c1 = aie::load_v<r * t>(pC + r * t);
    AIE_LOOP_MIN_ITERATION_COUNT(n / (2 * t))
    for (int blk = 0; blk < n / (2 * t); blk++) {
      CPtr pCb = pC + blk * (2 * r * t);
      CPtr pNext = (blk < n / (2 * t) - 1) ? pCb + 2 * r * t : pCb;
      MMUL acc0((aie::accum<accfloat, r * t>(c0)));
      MMUL acc1((aie::accum<accfloat, r * t>(c1)));
      c0 = aie::load_v<r * t>(pNext);
      c1 = aie::load_v<r * t>(pNext + r * t);
      AStream a_hoist(pA_in);
      a_hoist.seek(rb * (k / s));
      AStream a_pop(pA_in);
      a_pop.seek(rb * (k / s) + HOIST + z * blk);
#pragma unroll
      for (int q = 0; q < k / s; q++) {
        aie::block_vector<bfp16ebs8, r * s> A0 = (q < HOIST) ? a_hoist.pop() : a_pop.pop();
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
  return;
#endif
#ifdef BLK12S
  // BLK12 with each accumulator split over even / odd k-subs: four independent
  // vmac chains of 4 instead of two of 8, so the ~3-cycle MAC latency is hidden
  // (Peano's BLK12 loop was 27 bundles for 16 MACs because of that chain).
  // The halves are summed once per block (fp32 adds; reassociation only).
  for (int rb = 0; rb < m_a / r; rb++) {
    BStream b_stream(pB_in);
    CPtr pC = (CPtr)acc_rows(c_out, stripe * m_a + rb * r);
    aie::vector<float, r * t> c0 = aie::load_v<r * t>(pC);
    aie::vector<float, r * t> c1 = aie::load_v<r * t>(pC + r * t);
    AIE_LOOP_MIN_ITERATION_COUNT(n / (2 * t))
    for (int blk = 0; blk < n / (2 * t); blk++) {
      CPtr pCb = pC + blk * (2 * r * t);
      CPtr pNext = (blk < n / (2 * t) - 1) ? pCb + 2 * r * t : pCb;
      MMUL acc0((aie::accum<accfloat, r * t>(c0)));
      MMUL acc1((aie::accum<accfloat, r * t>(c1)));
      MMUL acc2, acc3;  // odd k-subs, start from zero
      c0 = aie::load_v<r * t>(pNext);
      c1 = aie::load_v<r * t>(pNext + r * t);
      AStream a_stream(pA_in);
      a_stream.seek(rb * (k / s) + z * blk);
#pragma unroll
      for (int q = 0; q < k / s; q += 2) {
        aie::block_vector<bfp16ebs8, r * s> A0 = a_stream.pop();
        aie::block_vector<bfp16ebs8, s * t> B0 = b_stream.pop();
        aie::block_vector<bfp16ebs8, s * t> B1 = b_stream.pop();
        aie::block_vector<bfp16ebs8, r * s> A1 = a_stream.pop();
        aie::block_vector<bfp16ebs8, s * t> B2 = b_stream.pop();
        aie::block_vector<bfp16ebs8, s * t> B3 = b_stream.pop();
        acc0.mac(A0, aie::op_transpose(B0));
        acc1.mac(A0, aie::op_transpose(B1));
        if (q == 0) {
          acc2.mul(A1, aie::op_transpose(B2));
          acc3.mul(A1, aie::op_transpose(B3));
        } else {
          acc2.mac(A1, aie::op_transpose(B2));
          acc3.mac(A1, aie::op_transpose(B3));
        }
      }
      aie::store_v(pCb, aie::add(acc0.template to_vector<float>(), acc2.template to_vector<float>()));
      aie::store_v(pCb + r * t, aie::add(acc1.template to_vector<float>(), acc3.template to_vector<float>()));
    }
  }
  event1();
  return;
#endif
#ifdef K3
  // 2x2 block (16 rows x 16 cols) with the first row-half's 8 A panels
  // loop-invariant (hoisted into ex registers, 12 exist) and the second half's
  // popped behind an opaque seek. Per block: 16 acc loads + 8 A pops on unit A,
  // 16 B pops on unit B, 32 MACs. A packed row-half-major (pack.py half=True).
  for (int rb = 0; rb < m_a / (2 * r); rb++) {
    BStream b_stream(pB_in);
    CPtr pC = (CPtr)acc_rows(c_out, stripe * m_a + rb * 2 * r);
#if PIPE == 1
    AIE_PREPARE_FOR_POSTPIPELINING
#endif
    AIE_LOOP_MIN_ITERATION_COUNT(n / (2 * t))
    for (int blk = 0; blk < n / (2 * t); blk++) {
      CPtr pCb = pC + blk * (2 * r * t);
      MMUL acc0((aie::accum<accfloat, r * t>(aie::load_v<r * t>(pCb))));
      MMUL acc1((aie::accum<accfloat, r * t>(aie::load_v<r * t>(pCb + r * t))));
      MMUL acc2((aie::accum<accfloat, r * t>(aie::load_v<r * t>(pCb + r * n))));
      MMUL acc3((aie::accum<accfloat, r * t>(aie::load_v<r * t>(pCb + r * t + r * n))));
      AStream a_hoist(pA_in);
      a_hoist.seek(rb * (2 * k / s));  // invariant across blk: the compiler keeps these 8 in registers
      AStream a_pop(pA_in);
      a_pop.seek(rb * (2 * k / s) + k / s + z * blk);
#pragma unroll
      for (int q = 0; q < k / s; q++) {
        aie::block_vector<bfp16ebs8, r * s> A0 = a_hoist.pop();
        aie::block_vector<bfp16ebs8, r * s> A1 = a_pop.pop();
        aie::block_vector<bfp16ebs8, s * t> B0 = b_stream.pop();
        aie::block_vector<bfp16ebs8, s * t> B1 = b_stream.pop();
        acc0.mac(A0, aie::op_transpose(B0));
        acc1.mac(A0, aie::op_transpose(B1));
        acc2.mac(A1, aie::op_transpose(B0));
        acc3.mac(A1, aie::op_transpose(B1));
      }
      aie::store_v(pCb, acc0.template to_vector<float>());
      aie::store_v(pCb + r * t, acc1.template to_vector<float>());
      aie::store_v(pCb + r * n, acc2.template to_vector<float>());
      aie::store_v(pCb + r * t + r * n, acc3.template to_vector<float>());
    }
  }
  event1();
  return;
#endif
#ifdef K2
  // 2x2 register block (16 rows x 16 cols) over TWO k-steps per call: the A
  // argument holds two consecutive k-step sub-tiles of the same 16 rows
  // (pack.py k2=True), pB_in the first B tile and pB2 the second (same L1
  // object order). Per block: 16 acc loads, 32 A pops, 32 B pops, 64 MACs.
  {
    bfp16ebs8 *pB2 = pB_in + k * n * 9 / 8;  // second k-step's B tile follows in L1 (depth-3 ring, 2 acquired)
    for (int br = 0; br < m_a / (2 * r); br++) {
      BStream b_stream(pB_in);
      BStream b2_stream(pB2);
      CPtr pC = (CPtr)acc_rows(c_out, stripe * m_a + br * 2 * r);
      AIE_PREPARE_FOR_POSTPIPELINING
      AIE_LOOP_MIN_ITERATION_COUNT(n / (2 * t))
      for (int blk = 0; blk < n / (2 * t); blk++) {
        CPtr pCb = pC + blk * (2 * r * t);
        MMUL acc0((aie::accum<accfloat, r * t>(aie::load_v<r * t>(pCb))));
        MMUL acc1((aie::accum<accfloat, r * t>(aie::load_v<r * t>(pCb + r * t))));
        MMUL acc2((aie::accum<accfloat, r * t>(aie::load_v<r * t>(pCb + r * n))));
        MMUL acc3((aie::accum<accfloat, r * t>(aie::load_v<r * t>(pCb + r * t + r * n))));
        AStream a_stream(pA_in);
        a_stream.seek(br * (4 * k / s) + z * blk);
#pragma unroll
        for (int q = 0; q < k / s; q++) {
          aie::block_vector<bfp16ebs8, r * s> A0 = a_stream.pop();
          aie::block_vector<bfp16ebs8, r * s> A1 = a_stream.pop();
          aie::block_vector<bfp16ebs8, s * t> B0 = b_stream.pop();
          aie::block_vector<bfp16ebs8, s * t> B1 = b_stream.pop();
          acc0.mac(A0, aie::op_transpose(B0));
          acc1.mac(A0, aie::op_transpose(B1));
          acc2.mac(A1, aie::op_transpose(B0));
          acc3.mac(A1, aie::op_transpose(B1));
        }
#pragma unroll
        for (int q = 0; q < k / s; q++) {
          aie::block_vector<bfp16ebs8, r * s> A0 = a_stream.pop();
          aie::block_vector<bfp16ebs8, r * s> A1 = a_stream.pop();
          aie::block_vector<bfp16ebs8, s * t> B0 = b2_stream.pop();
          aie::block_vector<bfp16ebs8, s * t> B1 = b2_stream.pop();
          acc0.mac(A0, aie::op_transpose(B0));
          acc1.mac(A0, aie::op_transpose(B1));
          acc2.mac(A1, aie::op_transpose(B0));
          acc3.mac(A1, aie::op_transpose(B1));
        }
        aie::store_v(pCb, acc0.template to_vector<float>());
        aie::store_v(pCb + r * t, acc1.template to_vector<float>());
        aie::store_v(pCb + r * n, acc2.template to_vector<float>());
        aie::store_v(pCb + r * t + r * n, acc3.template to_vector<float>());
      }
    }
  }
  event1();
  return;
#endif
#ifdef BLK12
  // 1x2 register block (8 rows x 16 cols): 2 live accumulators + 2 prefetched
  // for the next block = 4 of the 5 dm registers, so the next block's fp32 C
  // loads overlap this block's MACs. Per block: 8 acc loads + 8 A pops on load
  // unit A, 16 B pops on unit B, 16 MACs. Needs A packed row-half-major
  // ([2][k/8][8 rows] per 16-row group: pack.py a_order="half").
  for (int rb = 0; rb < m_a / r; rb++) {
    BStream b_stream(pB_in);
    CPtr pC = (CPtr)acc_rows(c_out, stripe * m_a + rb * r);
    aie::vector<float, r * t> c0 = aie::load_v<r * t>(pC);
    aie::vector<float, r * t> c1 = aie::load_v<r * t>(pC + r * t);
    AIE_PREPARE_FOR_POSTPIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(n / (2 * t))
#ifdef BLKU
#pragma unroll BLKU
#endif
    for (int blk = 0; blk < n / (2 * t); blk++) {
      CPtr pCb = pC + blk * (2 * r * t);
      CPtr pNext = (blk < n / (2 * t) - 1) ? pCb + 2 * r * t : pCb;
      MMUL acc0((aie::accum<accfloat, r * t>(c0)));
      MMUL acc1((aie::accum<accfloat, r * t>(c1)));
      c0 = aie::load_v<r * t>(pNext);
      c1 = aie::load_v<r * t>(pNext + r * t);
      AStream a_stream(pA_in);
      a_stream.seek(rb * (k / s) + z * blk);
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
  return;
#endif
  for (int br = 0; br < m_a / (2 * r); br++) {
    BStream b_stream(pB_in);
    CPtr pC = (CPtr)acc_rows(c_out, stripe * m_a + br * 2 * r);

#ifndef NO_PREFETCH
    aie::vector<float, r * t> c0 = aie::load_v<r * t>(pC);
    aie::vector<float, r * t> c1 = aie::load_v<r * t>(pC + r * t);
#endif
#if PIPE == 1
    AIE_PREPARE_FOR_POSTPIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(n / (2 * t))
#elif PIPE == 2
    AIE_TRY_INITIATION_INTERVAL(PIPE_II)
    AIE_LOOP_MIN_ITERATION_COUNT(n / (2 * t))
#elif PIPE == 3
    AIE_LOOP_MIN_ITERATION_COUNT(n / (2 * t))
#elif PIPE == 4
    AIE_LOOP_UNROLL_FULL
#elif PIPE == 5
    AIE_LOOP_UNROLL(2)
#endif
    for (int blk = 0; blk < n / (2 * t); blk++) {
      CPtr pCb = pC + blk * (2 * r * t);
#ifndef NO_PREFETCH
      CPtr pNext = (blk < n / (2 * t) - 1) ? pCb + 2 * r * t : pCb;
      MMUL acc0((aie::accum<accfloat, r * t>(c0)));
      MMUL acc1((aie::accum<accfloat, r * t>(c1)));
#else
      MMUL acc0((aie::accum<accfloat, r * t>(aie::load_v<r * t>(pCb))));
      MMUL acc1((aie::accum<accfloat, r * t>(aie::load_v<r * t>(pCb + r * t))));
#endif
      MMUL acc2((aie::accum<accfloat, r * t>(aie::load_v<r * t>(pCb + r * n))));
      MMUL acc3(
          (aie::accum<accfloat, r * t>(aie::load_v<r * t>(pCb + r * t + r * n))));
#ifndef NO_PREFETCH
      c0 = aie::load_v<r * t>(pNext);
      c1 = aie::load_v<r * t>(pNext + r * t);
#endif

      AStream a_stream(pA_in);
      a_stream.seek(br * (2 * k / s) + z * blk);

      auto kstep = [&]<bool BFILL>() __attribute__((always_inline)) {
        aie::block_vector<bfp16ebs8, r * s> A0, A1;
        aie::block_vector<bfp16ebs8, s * t> B0, B1;
        if constexpr (BFILL) {
          B0 = b_stream.pop();
          B1 = b_stream.pop();
          A0 = a_stream.pop();
          A1 = a_stream.pop();
        } else {
          A0 = a_stream.pop();
          A1 = a_stream.pop();
          B0 = b_stream.pop();
          B1 = b_stream.pop();
        }
        acc0.mac(A0, aie::op_transpose(B0));
        acc1.mac(A0, aie::op_transpose(B1));
        acc2.mac(A1, aie::op_transpose(B0));
        acc3.mac(A1, aie::op_transpose(B1));
      };
#pragma unroll
      for (int q = 0; q < k / 32; q++) {
        kstep.template operator()<true>();
        kstep.template operator()<false>();
        kstep.template operator()<false>();
        kstep.template operator()<false>();
      }

      aie::store_v(pCb, acc0.template to_vector<float>());
      aie::store_v(pCb + r * t, acc1.template to_vector<float>());
      aie::store_v(pCb + r * n, acc2.template to_vector<float>());
      aie::store_v(pCb + r * t + r * n, acc3.template to_vector<float>());
    }
  }
  event1();
}

// fp32 accumulator -> bf16 output tile (round to nearest even), same blocked
// layout. Fused epilogues (bias, GELU, residual, bfp16 re-pack) go here.
void acc_store(bfloat16 *c_out) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr int half = (m / 2) * n;
  // In place: element i (fp32 at byte 4i) -> bf16 at byte 2i; each 64-wide
  // load happens before the store that could overlap it.
  float *src = (float *)c_out;
  for (int i = 0; i < half / 64; i++) {
    aie::vector<float, 64> v = aie::load_v<64>(src + 64 * i);
    aie::store_v(c_out + 64 * i, aie::accum<accfloat, 64>(v).to_vector<bfloat16>());
  }
  for (int i = 0; i < half / 64; i++) {
    aie::vector<float, 64> v = aie::load_v<64>(acc_hi + 64 * i);
    aie::store_v(c_out + half + 64 * i,
                 aie::accum<accfloat, 64>(v).to_vector<bfloat16>());
  }
}
}
