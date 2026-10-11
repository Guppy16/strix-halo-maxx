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

constexpr int SLOT_B = 768;
constexpr int IN_B = SLOT_B * EW_IN_SLOTS;
constexpr int OUT_B = SLOT_B * EW_OUT_SLOTS;
constexpr int D = 768;
constexpr int QS = D / 64;   // 12 k-steps per row
constexpr int KB = 1024;     // one k-step piece of 8 rows
constexpr int PB = 72;       // bytes per bfp panel
constexpr int PANB = 8 * PB; // 8 panels = one k-step's companion
constexpr int RBN = 18;      // row blocks per column

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

static inline void lacq(const void *p, int id, int n) { acquire_greater_equal(p, id + 48, n); }
static inline void lrel(void *p, int id, int n) { release(p, id + 48, n); }

static __attribute__((noinline)) void in_need(const uint32_t *ring, int bytes) {  // make `bytes` of unread input resident
  int n = 0;
  while (g_in_avail < bytes) {  // (no division: __divsi3 costs program memory)
    g_in_avail += SLOT_B;
    n++;
  }
  if (n) lacq(ring, L_IN_FILL, n);
}

static __attribute__((noinline)) void in_done(uint32_t *ring, int bytes) {  // consume `bytes`, release whole slots
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
static __attribute__((noinline)) B64 rows8(const bfloat16 *v8) {  // 8 per-row scalars -> [row][8 lanes]
  return aie::concat(aie::concat(aie::broadcast<bfloat16, 8>(v8[0]), aie::broadcast<bfloat16, 8>(v8[1]),
                                 aie::broadcast<bfloat16, 8>(v8[2]), aie::broadcast<bfloat16, 8>(v8[3])),
                     aie::concat(aie::broadcast<bfloat16, 8>(v8[4]), aie::broadcast<bfloat16, 8>(v8[5]),
                                 aie::broadcast<bfloat16, 8>(v8[6]), aie::broadcast<bfloat16, 8>(v8[7])));
}
static __attribute__((noinline)) F32 rows4f(const float *v4) {
  return aie::concat(aie::concat(aie::broadcast<float, 8>(v4[0]), aie::broadcast<float, 8>(v4[1])),
                     aie::concat(aie::broadcast<float, 8>(v4[2]), aie::broadcast<float, 8>(v4[3])));
}
static inline B32 rep4(const bfloat16 *v8) {  // 8 per-column values repeated for 4 rows
  B8 s = aie::load_v<8>(v8);
  return aie::concat(s, s, s, s);
}

// 1/sqrt(x) on 16 fp32 lanes: bit-trick seed + 3 Newton steps (fp32 vector math is emulated; ~300 bundles)
static __attribute__((noinline)) F16 rsqrt16(F16 x) {  // emulated fp32: ~4 KB of code, keep one copy
  I16 u = x.template cast_to<int32_t>();
  u = aie::sub(aie::broadcast<int32_t, 16>(0x5f3759df), aie::downshift(u, 1));
  F16 y = u.template cast_to<float>();
  const F16 c15 = aie::broadcast<float, 16>(1.5f), c05 = aie::broadcast<float, 16>(0.5f);
  AIE_LOOP_NO_UNROLL  // once per 8-row chunk: one copy of the emulated fp32 body saves program memory
  for (int i = 0; i < 3; i++) {
    F16 hx = FV(aie::mul(x, c05));
    F16 yy = FV(aie::mul(y, y));
    F16 t = aie::sub(c15, FV(aie::mul(hx, yy)));
    y = FV(aie::mul(y, t));
  }
  return y;
}

static inline void split32(const A32 &a, B32 &hi, B32 &lo) {  // fp32 accumulator -> bf16 hi + lo
  hi = BF(a);
  lo = BF(aie::mac(a, hi, (bfloat16)-1.0f));
}

// one static arena shared by the kernels (never two ops live at once): norm = stage 12,288 + params 3072;
// patch = 3 weight tiles 13,824 + 24 A panels 1728 + bias table 1024; ssm = see ew_ssm
alignas(64) static uint8_t g_arena[24576];
static bfloat16 *const g_stage = (bfloat16 *)g_arena;          // one chunk's X' (8 rows x 768) in k-step pieces
static bfloat16 *const g_p0 = (bfloat16 *)(g_arena + 12288);   // norm params: (scale + 1) | gamma
static bfloat16 *const g_p1 = (bfloat16 *)(g_arena + 13824);   // beta
alignas(64) static bfp16ebs8 g_panels[PANB];
alignas(64) static float g_sums[16], g_sq[16];
alignas(64) static bfloat16 g_rstd[16];
alignas(64) static float g_nb[8];        // LN: -mean * rstd per row
alignas(64) static bfloat16 g_rh[8], g_rl[8];  // LN: rstd hi / lo per row

extern "C" {

// X' = bf16(sum of n_in Q16 streams); mode 0: panels of X'; 1: RMSNorm (torch bf16 numerics) -> panels;
// 2: LayerNorm -> panels; 3: LayerNorm -> the X' part (the encoder norm). Params: a header slot (word 0 = the
// number of input streams) + 2 slots (mode 0/1) or 4 (2/3).
void ew_norm(uint32_t *in_ring, uint32_t *out_ring, int32_t mode) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const B32 ones = aie::broadcast<bfloat16, 32>(1.0f);
  const int pbytes = mode >= 2 ? 5 * SLOT_B : 3 * SLOT_B;  // [header slot: n_in][scale | gamma][beta]
  in_need(in_ring, pbytes);
  const int n_in = *(const int32_t *)in_at(in_ring, 0);
  in_copy(in_ring, g_p0, SLOT_B, D * 2);
  if (mode >= 2) in_copy(in_ring, g_p1, 3 * SLOT_B, D * 2);
  in_done(in_ring, pbytes);
  const int unit = n_in * KB;
  const F16 inv_d = aie::broadcast<float, 16>(1.0f / D);
  const B16 eps_b = aie::broadcast<bfloat16, 16>(1e-6f);
  const F16 eps_f = aie::broadcast<float, 16>(1e-6f);

  for (int rb = 0; rb < RBN; rb++) {
    // pass 1: X' = bf16(sum of the streams) into the stage; row sums (LN) / sums of bf16(X'^2) (RMS)
    A32 s0 = aie::zeros<accfloat, 32>(), s1 = s0, q0 = s0, q1 = s0;
    for (int q = 0; q < QS; q++) {
      in_need(in_ring, unit);
      for (int cb = 0; cb < 8; cb++) {
        const bfloat16 *p = in_at(in_ring, cb * 128);
        A32 a0 = aie::mul(aie::load_v<32>(p), ones), a1 = aie::mul(aie::load_v<32>(p + 32), ones);
        for (int i = 1; i < n_in; i++) {
          const bfloat16 *pi = in_at(in_ring, i * KB + cb * 128);
          a0 = aie::mac(a0, aie::load_v<32>(pi), ones);
          a1 = aie::mac(a1, aie::load_v<32>(pi + 32), ones);
        }
        B32 x0 = BF(a0), x1 = BF(a1);
        aie::store_v(g_stage + q * 512 + cb * 64, x0);
        aie::store_v(g_stage + q * 512 + cb * 64 + 32, x1);
        if (mode == 1) {
          q0 = aie::mac(q0, BF(aie::mul(x0, x0)), ones);
          q1 = aie::mac(q1, BF(aie::mul(x1, x1)), ones);
        } else if (mode >= 2) {
          s0 = aie::mac(s0, x0, ones);
          s1 = aie::mac(s1, x1, ones);
          q0 = aie::mac(q0, x0, x0);  // exact fp32 squares of bf16 values
          q1 = aie::mac(q1, x1, x1);
        }
      }
      in_done(in_ring, unit);
    }
    // per-row statistics on 16 fp32 lanes (rows 0..7 real, 8..15 padding)
    B64 rs = aie::zeros<bfloat16, 64>();
    if (mode == 1) {
      F32 f0 = FV(q0), f1 = FV(q1);
      for (int r = 0; r < 4; r++) {
        g_sums[r] = aie::reduce_add(f0.template extract<8>(r));
        g_sums[4 + r] = aie::reduce_add(f1.template extract<8>(r));
      }
      for (int r = 8; r < 16; r++) g_sums[r] = 1.0f;
      F16 sv = aie::load_v<16>(g_sums);
      B16 mean_b = BF(A16(FV(aie::mul(sv, inv_d))));
      B16 ve_b = aie::add(mean_b, eps_b);
      B16 rstd_b = BF(A16(rsqrt16(FV(A16(ve_b)))));
      aie::store_v(g_rstd, rstd_b);
      rs = rows8(g_rstd);
    } else if (mode >= 2) {
      F32 f0 = FV(s0), f1 = FV(s1), g0 = FV(q0), g1 = FV(q1);
      for (int r = 0; r < 4; r++) {
        g_sums[r] = aie::reduce_add(f0.template extract<8>(r));
        g_sums[4 + r] = aie::reduce_add(f1.template extract<8>(r));
        g_sq[r] = aie::reduce_add(g0.template extract<8>(r));
        g_sq[4 + r] = aie::reduce_add(g1.template extract<8>(r));
      }
      for (int r = 8; r < 16; r++) { g_sums[r] = 0.0f; g_sq[r] = 1.0f; }
      // var = E[x^2] - mean^2 in fp32 (16 emulated lanes, once per chunk); rstd = 1/sqrt(var + eps)
      F16 mean = FV(aie::mul(aie::load_v<16>(g_sums), inv_d));
      F16 var = aie::sub(FV(aie::mul(aie::load_v<16>(g_sq), inv_d)), FV(aie::mul(mean, mean)));
      F16 rstd = rsqrt16(aie::add(aie::max(var, aie::zeros<float, 16>()), eps_f));
      alignas(64) float rv[16], nbv[16];
      aie::store_v(rv, rstd);
      aie::store_v(nbv, FV(aie::mul(aie::neg(mean), rstd)));  // -mean * rstd
      for (int r = 0; r < 8; r++) {
        g_nb[r] = nbv[r];
        g_rh[r] = (bfloat16)rv[r];
        g_rl[r] = (bfloat16)(rv[r] - (float)g_rh[r]);
      }
    }
    // pass 2: emit per k-step [X' 1024 B | panels 576 B]
    B32 rs0 = rs.template extract<32>(0), rs1 = rs.template extract<32>(1);  // (mode 0 'none' is not supported)
    B64 rh = rows8(g_rh), rl = rows8(g_rl);
    B32 rh0 = rh.template extract<32>(0), rh1 = rh.template extract<32>(1);
    B32 rl0 = rl.template extract<32>(0), rl1 = rl.template extract<32>(1);
    F32 nb0 = rows4f(g_nb), nb1 = rows4f(g_nb + 4);
    for (int q = 0; q < QS; q++) {
      OutS ps(g_panels);
      for (int cb = 0; cb < 8; cb++) {
        bfloat16 *p = g_stage + q * 512 + cb * 64;
        B32 x0 = aie::load_v<32>(p), x1 = aie::load_v<32>(p + 32);
        B32 z0, z1;
        if (mode == 1) {
          B32 sc = rep4(g_p0 + q * 64 + cb * 8);
          z0 = BF(aie::mul(BF(aie::mul(x0, rs0)), sc));
          z1 = BF(aie::mul(BF(aie::mul(x1, rs1)), sc));
        } else if (mode >= 2) {
          B32 ga = rep4(g_p0 + q * 64 + cb * 8), be = rep4(g_p1 + q * 64 + cb * 8);
          A32 t0(nb0), t1(nb1);
          t0 = aie::mac(t0, x0, rh0);
          t0 = aie::mac(t0, x0, rl0);
          t1 = aie::mac(t1, x1, rh1);
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
        }
        if (mode == 3) {  // the encoder norm: the LN result replaces X' in the output
          aie::store_v(p, z0);
          aie::store_v(p + 32, z1);
        }
        ps.push(A64(aie::concat(z0, z1)).template to_vector<bfp16ebs8>());
      }
      out_bytes(out_ring, g_stage + q * 512, KB);
      out_bytes(out_ring, g_panels, PANB);
    }
  }
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
static bfloat16 *const g_hb = (bfloat16 *)(g_arena + 16448);      // bf16(h) [cb][8][8] (1024 B)
// per-head tables, [cb] -> the 8 per-column values broadcast over 4 rows (B32) or 8 rows (B64)
static B32 *const g_tcw = (B32 *)(g_arena + 17472);               // conv taps [t 4][cb 8] (2048 B)
static B32 *const g_tcb = (B32 *)(g_arena + 19520);               // conv bias [cb] (512 B)
static B32 *const g_tsp = (B32 *)(g_arena + 20032);               // softplus(a_param) [cb] (512 B)
static B64 *const g_tbx = (B64 *)(g_arena + 20544);               // input-gate bias [cb] (1024 B)
static B64 *const g_tba = (B64 *)(g_arena + 21568);               // a-gate bias [cb] (1024 B)
static bfloat16 *const g_e2 = (bfloat16 *)(g_arena + 22592);      // exp(2 log_a) -> sqrt(1 - .) (1024 B)
static_assert(22592 + 1024 <= sizeof(g_arena));

// the SSM's 6 KB output per chunk is exactly 8 output slots and always starts on a slot: reserve them once, store
// the 96 vectors straight into the ring, release them once
static inline bfloat16 *out_at(uint32_t *ring, int off) {
  int p = g_out_b + off;
  if (p >= OUT_B) p -= OUT_B;
  return (bfloat16 *)((char *)ring + p);
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
  for (int q = 0; q < QS; q++) {
    in_need(in_ring, SSM_PARAM_B);
    const int first = *(const int32_t *)in_at(in_ring, 0);
    in_copy(in_ring, g_sv, 0, 1536);
    in_copy(in_ring, g_wx, 1536, 4608);
    in_copy(in_ring, g_wa, 1536 + 4608, 4608);
    in_done(in_ring, SSM_PARAM_B);
    const B32 mfirst = aie::broadcast<bfloat16, 32>(first ? 1.0f : 0.0f);
    const B32 mrest = aie::broadcast<bfloat16, 32>(first ? 0.0f : 1.0f);
    ssm_tables();
    for (int rb = 0; rb < RBN; rb++) {
      in_need(in_ring, 7 * KB);
#ifdef SSM_COPY_ONLY  // timing differential: the ring traffic alone (6 KB out per 7 KB in)
      lacq(out_ring, L_OUT_FREE, 8);
      for (int i = 0; i < 6 * KB / 64; i++) aie::store_v(out_at(out_ring, 64 * i), aie::load_v<32>(in_at(in_ring, 64 * i)));
      lrel(out_ring, L_OUT_FILL, 8);
      g_out_b += 6 * KB;
      if (g_out_b >= OUT_B) g_out_b -= OUT_B;
      in_done(in_ring, 7 * KB);
      continue;
#endif
      // conv1d: xc = bf16(bf16(sum_t bf16(x_t w_t)) + b), taps x-3 (part 4) .. x (part 1)
      AIE_LOOP_NO_UNROLL
      for (int i = 0; i < 16; i++) {
        const int cb = i / 2, off = cb * 128 + (i % 2) * 64;
        A32 s = aie::mul(BF(aie::mul(aie::load_v<32>(in_at(in_ring, 4 * KB + off)), g_tcw[0 * 8 + cb])), ones);
        s = aie::mac(s, BF(aie::mul(aie::load_v<32>(in_at(in_ring, 3 * KB + off)), g_tcw[1 * 8 + cb])), ones);
        s = aie::mac(s, BF(aie::mul(aie::load_v<32>(in_at(in_ring, 2 * KB + off)), g_tcw[2 * 8 + cb])), ones);
        s = aie::mac(s, BF(aie::mul(aie::load_v<32>(in_at(in_ring, 1 * KB + off)), g_tcw[3 * 8 + cb])), ones);
        aie::store_v(g_xc + i * 32, BF(aie::mac(A32(aie::mul(BF(s), ones)), g_tcb[cb], ones)));
      }
      {  // xc -> bfp panels for the gate GEMMs
        OutS ps(g_xcp);
        for (int cb = 0; cb < 8; cb++) ps.push(A64(aie::load_v<64>(g_xc + cb * 64)).template to_vector<bfp16ebs8>());
      }
      gemm8(g_xcp, g_wx, 1, g_tbx, g_gx, 1);
      gemm8(g_xcp, g_wa, 1, g_tba, g_ga, 1);
#ifndef SSM_NO_GATES
      nk8::sigmoid_n(g_gx, 32, kc);  // gate_x and gate_a (contiguous)
      AIE_LOOP_NO_UNROLL
      for (int i = 0; i < 16; i++) {  // log_a = bf16(bf16(-8 gate_a) sp); e2 = exp(2 log_a) needs its own exp (torch)
        B32 loga = BF(aie::mul(BF(aie::mul(aie::load_v<32>(g_ga + i * 32), (bfloat16)-8.0f)), g_tsp[i / 2]));
        aie::store_v(g_ga + i * 32, loga);
        aie::store_v(g_e2 + i * 32, BF(aie::mul(loga, (bfloat16)2.0f)));
      }
      nk8::exp_n(g_ga, 16, kc);  // a
      nk8::exp_n(g_e2, 16, kc);  // e2
      AIE_LOOP_NO_UNROLL
      for (int i = 0; i < 16; i++) aie::store_v(g_e2 + i * 32, BF(aie::msc(A32(aie::mul(ones, ones)), aie::load_v<32>(g_e2 + i * 32), ones)));
      nk8::sqrt_n(g_e2, 16, kc);  // mult = sqrt(1 - e2)
#endif
      AIE_LOOP_NO_UNROLL
      for (int i = 0; i < 16; i++) {
        const int cb = i / 2, half = i % 2;
        B32 xc = aie::load_v<32>(g_xc + i * 32);
        B32 gx = aie::load_v<32>(g_gx + i * 32), a = aie::load_v<32>(g_ga + i * 32), mult = aie::load_v<32>(g_e2 + i * 32);
        B32 gated = BF(aie::mul(xc, gx));
        B32 nx = BF(aie::mul(gated, mult));
        // h = a h0 + nx (fp32; first frame: h = gated), hb = bf16(h)
        const float *h0p = (const float *)in_at(in_ring, (5 + half) * KB + cb * 128);
        A32 h0a(aie::load_v<32>(h0p));
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
      {
        OutS zs(g_panels);
        for (int cb = 0; cb < 8; cb++) {  // z = bf16(hb * y) -> panels
          B64 y = aie::load_v<64>(in_at(in_ring, cb * 128));
          B64 z = BF(aie::mul(aie::load_v<64>(g_hb + cb * 64), y));
          zs.push(A64(z).template to_vector<bfp16ebs8>());
        }
      }
      // output (8 slots): x, x-1, x-2 (the conv cache shifts by one frame), h rows 0-3, h rows 4-7, z panels + pad
      lacq(out_ring, L_OUT_FREE, 8);
      AIE_LOOP_NO_UNROLL
      for (int i = 0; i < 3 * KB / 64; i++) aie::store_v(out_at(out_ring, 64 * i), aie::load_v<32>(in_at(in_ring, KB + 64 * i)));
      AIE_LOOP_NO_UNROLL
      for (int i = 0; i < 2048 / 64; i++) aie::store_v(out_at(out_ring, 3 * KB + 64 * i), aie::load_v<32>((const bfloat16 *)g_hout + 32 * i));
      AIE_LOOP_NO_UNROLL
      for (int i = 0; i < PANB / 64; i++) aie::store_v(out_at(out_ring, 5 * KB + 64 * i), aie::load_v<32>((const bfloat16 *)g_panels + 32 * i));
      AIE_LOOP_NO_UNROLL
      for (int i = PANB / 64; i < KB / 64; i++) aie::store_v(out_at(out_ring, 5 * KB + 64 * i), aie::zeros<bfloat16, 32>());
      lrel(out_ring, L_OUT_FILL, 8);
      g_out_b += 6 * KB;
      if (g_out_b >= OUT_B) g_out_b -= OUT_B;
      in_done(in_ring, 7 * KB);
    }
  }
}
}
