// design7 elementwise kernels for the row-3 core of a column. One call = one whole op; the kernel drives the
// input / output rings itself (lock intrinsics, 768 B slots, wrap-around addressing per 64 B vector).
//
// Input ring: a shim -> core packet flow fills SLOT_B bytes per BD execution (one BD iterating over the ring);
// the kernel acquires L_IN_FILL by the slots it needs and releases L_IN_FREE when done. Output ring: the kernel
// acquires L_OUT_FREE per slot before writing it and releases L_OUT_FILL when the slot is full; the core -> shim
// BD sends slot by slot.
//
// Activations arrive in the blocked layout L: an 8-row chunk = [12 k-steps][8 col-blocks][8 rows][8 cols] bf16
// (12,288 B), so a 32-lane vector holds 4 rows x 8 columns and an 8x8 block is one bfp16ebs8 panel.
#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

constexpr int SLOT_B = 768;
constexpr int RING_SLOTS = EW_RING_SLOTS;
constexpr int RING_B = SLOT_B * RING_SLOTS;
constexpr int D = 768;          // model width
constexpr int CHUNK_B = 8 * D * 2;  // an 8-row bf16 chunk: 12,288 B = 16 slots
constexpr int CHUNK_SLOTS = CHUNK_B / SLOT_B;
constexpr int QS = D / 64;      // 12 k-steps per row
constexpr int PB = 72;          // bytes per bfp panel

using B32 = aie::vector<bfloat16, 32>;
using B64 = aie::vector<bfloat16, 64>;
using B16 = aie::vector<bfloat16, 16>;
using F16 = aie::vector<float, 16>;
using I16 = aie::vector<int32_t, 16>;
using A16 = aie::accum<accfloat, 16>;
using A32 = aie::accum<accfloat, 32>;
using A64 = aie::accum<accfloat, 64>;
using OutS = aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::none>;
#define BF(e) ((e).template to_vector<bfloat16>())
#define FV(e) ((e).template to_vector<float>())

static int g_in_b = 0;       // byte position of the next unread input slot
static int g_out_b = 0;      // byte position of the next output byte
static int g_out_slot_b = 0; // bytes written into the current output slot

static inline void lacq(const void *p, int id, int n) { acquire_greater_equal(p, id + 48, n); }
static inline void lrel(void *p, int id, int n) { release(p, id + 48, n); }

// pointer to byte offset `off` of the input region that starts at the ring's read position (wraps)
static inline const bfloat16 *in_at(const uint32_t *ring, int off) {
  int p = g_in_b + off;
  if (p >= RING_B) p -= RING_B;
  return (const bfloat16 *)((const char *)ring + p);
}

static inline void in_advance(int bytes) {
  g_in_b += bytes;
  if (g_in_b >= RING_B) g_in_b -= RING_B;
}

// append one 64 B vector to the output ring, handling the slot locks
static inline void out64(uint32_t *ring, const bfloat16 *src) {
  if (g_out_slot_b == 0) lacq(ring, L_OUT_FREE, 1);
  aie::store_v((bfloat16 *)((char *)ring + g_out_b), aie::load_v<32>(src));
  g_out_b += 64;
  g_out_slot_b += 64;
  if (g_out_slot_b == SLOT_B) {
    lrel(ring, L_OUT_FILL, 1);
    g_out_slot_b = 0;
  }
  if (g_out_b == RING_B) g_out_b = 0;
}

static inline B64 rows8(const bfloat16 *v8) {
  return aie::concat(aie::concat(aie::broadcast<bfloat16, 8>(v8[0]), aie::broadcast<bfloat16, 8>(v8[1]),
                                 aie::broadcast<bfloat16, 8>(v8[2]), aie::broadcast<bfloat16, 8>(v8[3])),
                     aie::concat(aie::broadcast<bfloat16, 8>(v8[4]), aie::broadcast<bfloat16, 8>(v8[5]),
                                 aie::broadcast<bfloat16, 8>(v8[6]), aie::broadcast<bfloat16, 8>(v8[7])));
}

// 1/sqrt(x) on 16 fp32 lanes: bit-trick seed + 3 Newton steps (fp32 vector math is emulated; ~300 bundles)
static inline F16 rsqrt16(F16 x) {
  I16 u = x.template cast_to<int32_t>();
  u = aie::sub(aie::broadcast<int32_t, 16>(0x5f3759df), aie::downshift(u, 1));
  F16 y = u.template cast_to<float>();
  const F16 c15 = aie::broadcast<float, 16>(1.5f), c05 = aie::broadcast<float, 16>(0.5f);
  for (int i = 0; i < 3; i++) {
    F16 hx = FV(aie::mul(x, c05));
    F16 yy = FV(aie::mul(y, y));
    F16 t = aie::sub(c15, FV(aie::mul(hx, yy)));
    y = FV(aie::mul(y, t));
  }
  return y;
}

alignas(64) static bfloat16 g_sc32[QS * 8][32];  // (scale + 1) per 8-column block, repeated for 4 rows
alignas(64) static bfloat16 g_rstd[16];
alignas(64) static float g_sums[16];
alignas(64) static bfp16ebs8 g_panels[8 * PB];  // one k-step's 8 panels (576 B, one element per byte) before they go to the ring

extern "C" {

// RMSNorm of n_chunks 8-row chunks: z = bf16(bf16(x * rstd) * (scale + 1)), rstd = bf16(rsqrt(bf16(bf16(mean
// bf16(x^2)) + eps))) (torch's bf16 numerics: fp32 reductions, bf16 after every op). Params first: 2 slots
// holding (scale + 1) as 768 bf16. Out: per chunk 12 x 576 B of bfp panels (9 slots).
void ew_rmsnorm(uint32_t *in_ring, uint32_t *out_ring, int32_t n_chunks) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  const B32 ones = aie::broadcast<bfloat16, 32>(1.0f);
  // params
  lacq(in_ring, L_IN_FILL, 2);
  for (int b = 0; b < QS * 8; b++) {
    B16 s16 = aie::load_v<16>(in_at(in_ring, (b / 2) * 32));
    aie::vector<bfloat16, 8> s8 = s16.template extract<8>(b % 2);
    aie::store_v(g_sc32[b], aie::concat(s8, s8, s8, s8));
  }
  lrel(in_ring, L_IN_FREE, 2);
  in_advance(2 * SLOT_B);
  const F16 inv_d = aie::broadcast<float, 16>(1.0f / D);
  const B16 eps = aie::broadcast<bfloat16, 16>(1e-6f);

  for (int ch = 0; ch < n_chunks; ch++) {
    lacq(in_ring, L_IN_FILL, CHUNK_SLOTS);
    // pass 1: sum of bf16(x^2) per row, rows 0-3 in a0's four 8-lane groups, rows 4-7 in a1's
    A32 a0 = aie::zeros<accfloat, 32>(), a1 = a0;
    for (int b = 0; b < QS * 8; b++) {
      const bfloat16 *p = in_at(in_ring, b * 128);
      B32 v0 = aie::load_v<32>(p), v1 = aie::load_v<32>(p + 32);
      a0 = aie::mac(a0, BF(aie::mul(v0, v0)), ones);
      a1 = aie::mac(a1, BF(aie::mul(v1, v1)), ones);
    }
    aie::vector<float, 32> f0 = FV(a0), f1 = FV(a1);
    for (int r = 0; r < 4; r++) {
      g_sums[r] = aie::reduce_add(f0.template extract<8>(r));
      g_sums[4 + r] = aie::reduce_add(f1.template extract<8>(r));
    }
    for (int r = 8; r < 16; r++) g_sums[r] = 1.0f;
    // per-row scalars on 16 fp32 lanes: mean -> bf16, + eps -> bf16, rsqrt -> bf16
    F16 sv = aie::load_v<16>(g_sums);
    B16 mean_b = BF(A16(FV(aie::mul(sv, inv_d))));
    B16 ve_b = aie::add(mean_b, eps);
    F16 ve = FV(A16(ve_b));
    B16 rstd_b = BF(A16(rsqrt16(ve)));
    aie::store_v(g_rstd, rstd_b);
    B64 rs = rows8(g_rstd);
    B32 rs0 = rs.template extract<32>(0), rs1 = rs.template extract<32>(1);
    // pass 2: normalise, scale, convert each 8x8 block to a bfp panel, 8 panels per k-step
    for (int q = 0; q < QS; q++) {
      OutS ps(g_panels);
      for (int cb = 0; cb < 8; cb++) {
        const bfloat16 *p = in_at(in_ring, (q * 8 + cb) * 128);
        B32 y0 = BF(aie::mul(aie::load_v<32>(p), rs0));
        B32 y1 = BF(aie::mul(aie::load_v<32>(p + 32), rs1));
        B32 sc = aie::load_v<32>(g_sc32[q * 8 + cb]);
        B32 z0 = BF(aie::mul(y0, sc)), z1 = BF(aie::mul(y1, sc));
        ps.push(A64(aie::concat(z0, z1)).template to_vector<bfp16ebs8>());
      }
      for (int i = 0; i < 8 * PB / 64; i++) out64(out_ring, (const bfloat16 *)(g_panels + 64 * i));
    }
    lrel(in_ring, L_IN_FREE, CHUNK_SLOTS);
    in_advance(CHUNK_B);
  }
  event1();
}
}
