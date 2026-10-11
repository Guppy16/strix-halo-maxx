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
#ifdef GELU_FLAT  // round 12: flat-edge tables (r12/lutflat.py: + 4 flat segments per side, bias 36; the C++ paths are unchanged)
#include "gelu_lut_flat.h"
#else
#include "gelu_lut.h"  // generated: python3 r6/lutgen.py -3 ../gelu_lut.h data,3 (ab copies in the data area, cd in bank 3)
#endif
#include "exp_lut.h"   // generated: python3 r7/explut.py (the attention exp; second copy written by attn_begin)

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
#ifdef MM_ASM
extern "C" void mm_core(const bfp16ebs8 *pA, const bfp16ebs8 *pB, float *pC);
#endif
#ifdef MM_PIECES  // round 9 (npu-research P20 stage a): r7/gen_mm.py --pieces
extern "C" void mm_core_p(const bfp16ebs8 *pA, const bfp16ebs8 *pB, float *piece0, int d1, int d2, int d3);
#endif
#ifdef MM_KS  // round 9 (P20 stage c): the whole k-step in asm, stripes pipelined (r9/gen_ks.py -> build/mm_ks.s / .h)
#include "build/mm_ks.h"
extern "C" void mm_ks(const int32_t *rec, const bfp16ebs8 *pB, float *piece0);
#ifndef KSC
static int32_t g_ks_tab[MM_KS_NREC];
static const bfloat16 *g_ks_for = nullptr;
#endif
#endif
#ifdef KSP38
static uint32_t g_kc[5];  // round 18: counters 0-2, timer, k-steps (summed over a tile's mm_ks calls; acc_epilogue writes them)
#endif
#ifdef PROF  // timing build (round 7, npu-research P15): core-timer stamps per tile, written over the tile's first words
// (the runtime sequence enables each core's processor bus first: read_tm hangs the core without it)
static inline __attribute__((always_inline)) uint32_t stamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}
static uint32_t g_p_t0, g_p_busy, g_p_calls;
// round 8: the gaps between calls (call + IRON lock code + waits): at k-step starts (the B acquire), inside k-steps
// (the A ring only), the tile's first wait, and the largest k-step-start gap
static uint32_t g_p_last, g_p_gkb, g_p_ga, g_p_first, g_p_gmax;
// P24: epilogue end -> the next acc_zero entry (C handoff + loop code), acc_zero's own cycles
static uint32_t g_p_epi_end, g_p_hand, g_p_zero;
// round 8: attention per head group: before begin (Q copies + A waits), begin, K phase, V phase, waits between calls, end
static uint32_t ga_last, ga_pre, ga_beg, ga_k, ga_v, ga_gap;
#endif

static inline float *acc_rows(bfloat16 *c_out, int row) {
  return row < ACC_LO ? (float *)c_out + row * n : acc_hi + (row - ACC_LO) * n;
}

#ifdef MM_PIECES
// Round 9 (npu-research P20 / ANALYSIS rev 4 §3.1): the accumulators of (row block rb, column pair) = 8 rows x 32 columns
// fp32 = column blocks 4 pair .. 4 pair + 3 at 64 floats each, are one 1 KB piece. A k-step walks the 36 pieces in the
// order stripe j, groups (2j, 0) (2j, 1) (2j+1, 0) (2j+1, 1); at every group boundary one piece's 16 stores co-issue
// with the next piece's 16 loads, so consecutive pieces must sit in different banks (two accesses to one bank in a
// bundle stall). Layout (affine, so the epilogue passes stay strided loops):
//   pair 0, rb < 12: C slot rb (C bytes [1 KB rb, +1 KB), bank 2; an epilogue writes row block rb's output at or below
//                    its own slot, after reading it)
//   pair 0, rb >= 12: acc_hi piece rb - 12 (0..5)       pair 1: acc_hi piece 6 + rb (6..23)
// Stripes 0-5 alternate C / acc_hi; stripes 6-8 alternate acc_hi pieces 0-5 and 18-23, which sit in different banks
// as long as acc_hi's bank boundary falls inside pieces 6-17 (acc_hi at 0x71400 in the tests, 0x71900 in the step).
static inline float *piece(bfloat16 *c_out, int rb, int pair) {
  return pair ? acc_hi + 256 * (6 + rb) : rb < 12 ? (float *)c_out + 256 * rb : acc_hi + 256 * (rb - 12);
}
// column block cb (64 fp32: 8 rows x 8) of row block rb
static inline float *acc_cb(bfloat16 *c_out, int rb, int cb) { return piece(c_out, rb, cb >> 2) + 64 * (cb & 3); }
// the compact bf16 of row block rb, half h (column blocks 4h..4h+3: 256 bf16) at the start of its piece
static inline bfloat16 *rb_bf2(bfloat16 *c_out, int rb, int h) { return (bfloat16 *)piece(c_out, rb, h); }
#else
static inline float *acc_cb(bfloat16 *c_out, int rb, int cb) { return acc_rows(c_out, rb * 8) + 64 * cb; }
static inline bfloat16 *rb_bf2(bfloat16 *c_out, int rb, int h) { return (bfloat16 *)acc_rows(c_out, rb * 8) + 256 * h; }
#endif

using nk8::ExpC;
extern "C" {

#ifdef ZASM  // round 14: r14/zero2.s, the two zero fills as zero-overhead loops (1 store per bundle, -~100 B of text)
}
extern "C" void zero2(float *a, float *b, int trips);  // a: 64 B per trip, b: 128 B per trip
extern "C" {
#endif
#ifdef ZHI  // round 19: acc_hi (24,576 B) is zeroed right after the C object's release (design8 calls acc_zero_hi there), which
// overlaps the C drain the next acc_zero waits for; acc_zero then zeroes only the C part unless something used acc_hi since
// (the second tile of a 2-tile object, the attention: attn_begin clears the flag)
static int32_t g_hi_clean = 0;
#ifdef CSPLIT  // round 19: the C object drains in two parts (r19/csplit.py patches each GEMM core's MM2S after IRON's lowering):
// bytes 0 .. 12,287 (the accumulators' C part) release the C prod lock, so the next tile starts once they are out; bytes 12,288 ..
// 18,431 drain during the next tile and then release lock CS_Z (init 1). Anything that writes part 2 takes CS_Z first, once per
// object (g_zh; cleared after the object's release: acc_zero_hi, attn_end). The core releases the C cons lock by 2 (patched).
static int32_t g_zh = 0;
static inline __attribute__((always_inline)) void cs_take(void *c) {
  if (!g_zh) {
    acquire_greater_equal(c, 48 + CS_Z, 1);
    g_zh = 1;
  }
}
#endif
void acc_zero_hi() {
  g_hi_clean = 1;
#ifdef CSPLIT
  g_zh = 0;
#endif
  zero2(acc_hi, acc_hi + 2048, 128);  // 8 KB at a + 16 KB at b (a tail call: no frame)
}
#endif
#if defined(CSPLIT) && !defined(ZHI)
#error "CSPLIT clears its per-object flag in acc_zero_hi (ZHI)"
#endif
void acc_zero(bfloat16 *c_out) {
#ifdef PROF
  const uint32_t tz = stamp();
#endif
#if defined(ZASM) && defined(ZHI) && !defined(PROF)
  static_assert(ACC_LO * n * 4 == 12288 && (m - ACC_LO) * n * 4 == 24576);
  {  // the C part only (12,288 B = 4 KB at a + 8 KB at b) when acc_hi is clean; flags first, zero2 as a tail call (no frame)
    const int32_t cl = g_hi_clean;
    g_hi_clean = 0;
    g_stripe = 0;
    zero2((float *)c_out, cl ? (float *)c_out + 1024 : acc_hi, cl ? 64 : ACC_LO * n * 4 / 64);
    return;
  }
#elif defined(ZASM) && defined(ZHI)
#error "ZHI under PROF: not wired"
#elif defined(ZASM)
  static_assert(ACC_LO * n * 4 % 64 == 0 && (m - ACC_LO) * n * 4 == 2 * ACC_LO * n * 4);
  zero2((float *)c_out, acc_hi, ACC_LO * n * 4 / 64);
#else
  const aie::vector<float, 32> z = aie::zeros<float, 32>();
  float *p0 = (float *)c_out;
#ifdef ATTK_COUNT
#pragma clang loop unroll(disable)
#endif
  for (int i = 0; i < ACC_LO * n / 32; i++) aie::store_v(p0 + 32 * i, z);
#ifdef ATTK_COUNT
#pragma clang loop unroll(disable)
#endif
  for (int i = 0; i < (m - ACC_LO) * n / 32; i++) aie::store_v(acc_hi + 32 * i, z);
#endif
  g_stripe = 0;
#ifdef PROF
  g_p_busy = g_p_calls = g_p_gkb = g_p_ga = g_p_first = g_p_gmax = 0;
  g_p_t0 = g_p_last = stamp();
  g_p_hand = tz - g_p_epi_end;  // (garbage for a core's first tile)
  g_p_zero = g_p_t0 - tz;
#endif
}

#if defined(MM_ASM) && defined(MM_KSTEP)
// round 9 (npu-research P20, stage b0): one call per k-step. The A ring's locks are pinned (design8 KSTEP: aF = 4 free,
// aU = 5 full, +48 = this tile) and taken here, so IRON's per-call lock code and call sequence run once per k-step.
// The 9 stripes cycle through the 3 slots; a slot is released once mm_core returned (its last pop has read memory).
#ifdef MM_KS
// The record stream mm_ks reads (gen_ks.py's order), built once per C object address. Round 13 (npu-research T1): every record
// is a compile-time constant plus b x (c_out - acc_hi) (the cursor jumps between pieces in the C object and in acc_hi), or an
// A slot address: the tables are constexpr, the builder ~60 B (the arithmetic form was 832 B of program memory).
struct KsTab {
  int16_t a[MM_KS_NREC];  // the constant (piece offsets < 32 KB), a slot index, a flag
  int8_t b[MM_KS_NREC];   // x (c_out - acc_hi) in bytes; 2 = slot[a] (L1 is as tight as program memory: 174 B of tables)
};
static constexpr KsTab ks_tab = [] {
  KsTab t{};
  // piece(rb, pair) as (region, byte offset): region 0 = c_out, 1 = acc_hi
  auto reg = [](int rb, int pair) { return pair ? 1 : rb < 12 ? 0 : 1; };
  auto off = [](int rb, int pair) { return pair ? 1024 * (6 + rb) : rb < 12 ? 1024 * rb : 1024 * (rb - 12); };
  for (int i = 0; i < MM_KS_NREC; i++) {
    const int s = mm_ks_order[i][0], f = mm_ks_order[i][1];
    t.b[i] = 0;
    if (f == 0) {
      t.b[i] = 2;
      t.a[i] = s < rho ? (s >= 6 ? s - 6 : s >= 3 ? s - 3 : s) : 0;  // the dummy stripe reads slot 0 (no acquire)
    } else if (f == 5) {
      t.a[i] = s < rho;  // acq.cond's flag: 0 skips the dummy stripe's acquire
    } else if (s < rho) {
      const int k = f - 1, rb = 2 * s + (k >> 1), pair = k & 1;
      const int prb = k ? 2 * s + ((k - 1) >> 1) : 2 * s - 1, ppair = k ? (k - 1) & 1 : 1;
      if (s == 0 && k == 0) {
        t.a[i] = MM_KS_P4_BACK;
      } else {
        t.a[i] = (int16_t)(off(rb, pair) - off(prb, ppair) - 1024);
        t.b[i] = (int8_t)((reg(rb, pair) == 0) - (reg(prb, ppair) == 0));
      }
    } else {
      t.a[i] = f == 1 ? -1024 : 0;  // the dummy: re-read the last piece
    }
  }
  return t;
}();
#ifdef KSC  // round 18 (research rev 13 KSC): every GEMM core has the same placement, so the records are build-time data. The
// addresses come in as -DKS_ACC_HI / -DKS_C / -DKS_A0..2 (design8 checks them against the linked elf and the lowered MLIR after the
// build and fails it if anything moved); mm_kstep is then one tail call (-~280 B of program memory)
struct KsRec {
  int32_t r[MM_KS_NREC];
};
static constexpr KsRec ks_rec = [] {
  KsRec t{};
  const int32_t d = (int32_t)KS_C - (int32_t)KS_ACC_HI, slot[3] = {(int32_t)KS_A0, (int32_t)KS_A1, (int32_t)KS_A2};
  for (int i = 0; i < MM_KS_NREC; i++) t.r[i] = ks_tab.b[i] == 2 ? slot[ks_tab.a[i]] : ks_tab.a[i] + ks_tab.b[i] * d;
  return t;
}();
#else
static void ks_records(bfp16ebs8 *const slot[3], bfloat16 *c_out) {
  const int32_t d = (int32_t)((const char *)c_out - (const char *)acc_hi);
#pragma clang loop unroll(disable)
  for (int i = 0; i < MM_KS_NREC; i++)
    g_ks_tab[i] = ks_tab.b[i] == 2 ? (int32_t)(uintptr_t)slot[ks_tab.a[i]] : ks_tab.a[i] + ks_tab.b[i] * d;
  g_ks_for = c_out;
}
#endif
#endif
#ifdef MM_ROT  // round 18 (research pass 14 ROT, E5): the k-step with a rotating accumulator (r18/mm_rot_g2f.s from npu-research r19/
// gen_rot.py; design8 copies it to build/mm_rot.s / .h / .inc); B packed in mm_rot_border's panel order (pack8 B_ROT)
#include "build/mm_rot.inc"
#endif
#if defined(P35) && defined(KSC)
// round 18 (research P35): a tile's k-steps in one call. The B objects are taken here by lock id (KS_BCONS / KS_BPROD, +48 = this
// tile) and address (KS_B0 / KS_B1, checked after the link by design8 ksc_check). The count per tile (K / 64) is even, so IRON's
// own B index (bias objects, attention) is where it was; acc_epilogue records which buffer the bias came from: the next k-step
// starts at the other one.
static const bfp16ebs8 *g_bnext = (const bfp16ebs8 *)KS_B0;
#ifdef TPROF  // round 19 timing build: per-tile stamps (r19/tprof.py): tile start, after the first k-step, k-steps end, epilogue
static inline __attribute__((always_inline)) uint32_t tst() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}
static uint32_t g_t0, g_t1, g_t2, g_tlast, g_tnks;
#endif
extern "C" void mm_ktile(bfp16ebs8 *a0, bfp16ebs8 *a1, bfp16ebs8 *a2, bfloat16 *c_out, int32_t nks) {
  (void)a0, (void)a1, (void)a2;
  const bfp16ebs8 *b = g_bnext;
  float *pc = piece(c_out, 0, 0);
#ifdef TPROF
  g_t0 = tst();
  g_tnks = nks;
#endif
#pragma clang loop unroll(disable)
  for (int k = 0; k < nks; k++) {
#ifdef TPROF
    if (k == 1) g_t1 = tst();
#endif
    acquire_greater_equal((void *)b, 48 + KS_BCONS, 1);
#ifdef MM_ROT
#ifdef KSP38  // round 18: the counters around mm_rot (r18/ksp38.py)
    __builtin_aie2p_sched_barrier();
    const uint32_t c0 = read_tm(0x31520), c1 = read_tm(0x31524), c2 = read_tm(0x31528), c3 = read_tm(0x340F8);
    __builtin_aie2p_sched_barrier();
#endif
    mm_rot(rot_rec.r, b, pc);
#ifdef KSP38
    __builtin_aie2p_sched_barrier();
    g_kc[0] += read_tm(0x31520) - c0;
    g_kc[1] += read_tm(0x31524) - c1;
    g_kc[2] += read_tm(0x31528) - c2;
    g_kc[3] += read_tm(0x340F8) - c3;
    g_kc[4]++;
    __builtin_aie2p_sched_barrier();
#endif
#else
    mm_ks(ks_rec.r, b, pc);
#endif
    release((void *)b, 48 + KS_BPROD, 1);
    b = (const bfp16ebs8 *)((uintptr_t)b ^ ((uintptr_t)KS_B0 ^ (uintptr_t)KS_B1));
  }
#ifdef TPROF
  g_t2 = tst();
#endif
}
#endif
void mm_kstep(bfp16ebs8 *a0, bfp16ebs8 *a1, bfp16ebs8 *a2, bfp16ebs8 *__restrict pB_in, bfloat16 *c_out) {
#ifdef PROF
  const uint32_t tp = stamp();
  {
    const uint32_t gap = tp - g_p_last;
    if (g_p_calls == 0) g_p_first = gap;
    else { g_p_gkb += gap; g_p_gmax = gap > g_p_gmax ? gap : g_p_gmax; }
  }
#endif
  bfp16ebs8 *const slot[3] = {a0, a1, a2};
#if defined(MM_ROT) && defined(KSC)  // round 18: mm_ks is not linked under MM_ROT
  (void)slot;
  rot_kstep(a0, a1, a2, pB_in, c_out);
#elif defined(MM_KS) && defined(KSC)
  (void)slot;
#ifdef KSP38  // round 18 (research rev 13 E3 / KS-P38): core performance counters 0-2 and the timer around mm_ks (design8 sets the events)
  __builtin_aie2p_sched_barrier();
  const uint32_t c0 = read_tm(0x31520), c1 = read_tm(0x31524), c2 = read_tm(0x31528), c3 = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
#endif
  mm_ks(ks_rec.r, pB_in, piece(c_out, 0, 0));
#ifdef KSP38
  __builtin_aie2p_sched_barrier();
  g_kc[0] += read_tm(0x31520) - c0;
  g_kc[1] += read_tm(0x31524) - c1;
  g_kc[2] += read_tm(0x31528) - c2;
  g_kc[3] += read_tm(0x340F8) - c3;
  g_kc[4]++;
  __builtin_aie2p_sched_barrier();
#endif
#elif defined(MM_KS)
  if (c_out != g_ks_for) ks_records(slot, c_out);  // once per C object address (round 13: a cold function, program memory)
  mm_ks(g_ks_tab, pB_in, piece(c_out, 0, 0));
#else
#pragma clang loop unroll(disable)
  for (int st = 0; st < rho; st += 3) {
#pragma unroll
    for (int j = 0; j < 3; j++) {
      acquire_greater_equal(slot[j], 48 + 5, 1);
#ifdef MM_PIECES
      {
        const int rb = 2 * (st + j);
        const char *p0 = (const char *)piece(c_out, rb, 0), *p1 = (const char *)piece(c_out, rb, 1);
        const char *p2 = (const char *)piece(c_out, rb + 1, 0), *p3 = (const char *)piece(c_out, rb + 1, 1);
        mm_core_p(slot[j], pB_in, (float *)p0, (int)(p1 - p0 - 1024), (int)(p2 - p1 - 1024), (int)(p3 - p2 - 1024));
      }
#else
      mm_core(slot[j], pB_in, acc_rows(c_out, (st + j) * m_a));
#endif
      release(slot[j], 48 + 4, 1);
    }
  }
#endif
#ifdef PROF
  g_p_last = stamp();
  g_p_busy += g_p_last - tp;
  g_p_calls += rho;
#endif
}
#endif

void mm_acc(bfp16ebs8 *__restrict pA_in, bfp16ebs8 *__restrict pB_in, bfloat16 *c_out) {
  event0();
#ifdef PROF
  const uint32_t tp = stamp();
  {
    const uint32_t gap = tp - g_p_last;
    if (g_p_calls == 0) g_p_first = gap;
    else if (g_stripe == 0) { g_p_gkb += gap; g_p_gmax = gap > g_p_gmax ? gap : g_p_gmax; }
    else g_p_ga += gap;
  }
#endif
  using AStream = InBufStream<bfp16ebs8, r * s, aie_dm_resource::a>;
  using BStream = InBufStream<bfp16ebs8, s * t, aie_dm_resource::b>;
  using CPtr = float __aie_dm_resource_c *__restrict;
  using MMUL = aie::mmul<r, s, t, bfp16ebs8>;
  const int stripe = g_stripe;
  g_stripe = (g_stripe == rho - 1) ? 0 : g_stripe + 1;
#ifdef MM_NOCOMPUTE  // npu-research P1 twin (timing only): locks and DMA as usual, no MACs
  event1();
  return;
#endif
#if defined(MM_ASM) && !defined(MM_PIECES)  // round 7: the call's 128 MACs hand-scheduled (r7/gen_mm.py; B in pack8's pair4 order), bit-identical
  mm_core(pA_in, pB_in, acc_rows(c_out, stripe * m_a));
#ifdef PROF
  g_p_last = stamp();
  g_p_busy += g_p_last - tp;
  g_p_calls++;
#endif
  event1();
  return;
#endif
#ifdef MM_TWICE  // npu-research P1 twin (timing only): 2x the MACs on the same A and B
  for (int twice = 0; twice < 2; twice++)
#endif
  // one A pass per 16-column block, the next block's accumulators prefetched. The hardware loop is 28 bundles per
  // 16 mmuls: Peano holds the 8 A panels in registers (hoisted out of the block loop), pops B on ldb (16 pops + 2
  // fills) and runs 2 accumulator chains at distance 3 (the bound: 3 bundles per 2 mmuls; npu-research rev 2).
  // Round 7 tried 4 chains (even / odd k): 54 bundles with spills, 42 without the c prefetch, 61 with A popped per
  // block (an opaque seek): Peano neither pipelines across blocks nor keeps pops in flight. Next: asm (P18). Round 6 tried
  // two column blocks per A pass and half-k passes holding A in registers: Peano then runs out of ex registers for
  // B pops in flight (pops land at +8) and the loops got slower (45-75 bundles per 16-32 mmuls): see NOTES 11.
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
#ifdef PROF
  g_p_last = stamp();
  g_p_busy += g_p_last - tp;
  g_p_calls++;
#endif
  event1();
}
}

// ---- epilogues (round 6: passes instead of per-block calls) -------------------------------------------------
// Round 5 ran every 64-element block through minsize helpers (blk_src / blk_bias / gelu32x2 calls, spills): ~77
// bundles per block before any GELU. Now one pipelined bias pass turns the fp32 tile into bf16, the GELU runs as an
// array pass over bf16 (nk8::gelu_n) and the packing pass reads bf16. The numerics are unchanged (bit-identical).
//
// Where the bf16 goes: flags 0/1 (in place) write block i at byte 128 i of the C object (the drained layout). The
// other flags write row block rb compactly into the first half of its own fp32 area (rb_bf): rows < ACC_LO stay in
// the C object's first 12,288 B, so the gated stash behind them and every packed output written behind the reads
// stay intact.

// bf16(acc + bias) for nrb row blocks: row block r's 8 fp32 blocks at src + 512 r, its bf16 blocks at dst + dstride r.
// In the C object src and dst are the same memory: the writes trail the reads (block i's bf16 lands at or below its
// own fp32), so __restrict is safe for any schedule that keeps each block's stores after its own loads.
static __attribute__((noinline)) void bias_pass(const float *__restrict src, bfloat16 *__restrict dst, const bfloat16 *__restrict bias,
                                                int nrb, int dstride) {
  const B32 ones = aie::broadcast<bfloat16, 32>(1.0f);
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int r = 0; r < nrb; r++) {
#pragma unroll
    for (int cb = 0; cb < 8; cb++) {
      A32 a0(aie::load_v<32>(src + 64 * cb)), a1(aie::load_v<32>(src + 64 * cb + 32));
      a0 = aie::mac(a0, aie::load_v<32>(bias + 64 * cb), ones);
      a1 = aie::mac(a1, aie::load_v<32>(bias + 64 * cb + 32), ones);
      aie::store_v(dst + 64 * cb, BF(a0));
      aie::store_v(dst + 64 * cb + 32, BF(a1));
    }
    src += 512;
    dst += dstride;
  }
}

#ifdef MM_PIECES
// bf16(acc + bias) for the pieces i = 2 rb + h, h0 <= h < h0 + nh: to_c: row block rb's half h at C byte 1024 rb + 512 h
// (the drained layout, flags 0/1), else in place at the piece's start (rb_bf2). In place the writes trail the reads
// within the piece (as bias_pass); to C, half 0 of every row block runs before half 1 (a separate call), whose bf16
// lands on half 0's fp32 blocks 2-3 in C slot rb: no two pieces of one call overlap, so __restrict holds per call.
// bf16(acc + bias) for npc pieces at src, src + 1 KB, ... (column half h: bias + 256 h) -> dst, dst + dstride, ...
// In place (dst = src) the writes trail the reads within a piece; the strided calls never overlap another call's unread
// piece (the drained layout's half 1 lands on half 0's fp32 blocks 2-3 in C slot rb, which an earlier call read).
static __attribute__((noinline)) void bias_pcs(const float *__restrict src, bfloat16 *__restrict dst, const bfloat16 *__restrict bias,
                                               int npc, int dstride) {
  const B32 ones = aie::broadcast<bfloat16, 32>(1.0f);
  AIE_LOOP_MIN_ITERATION_COUNT(3)
  for (int i = 0; i < npc; i += 2) {  // two pieces per trip (npc even): 16 independent vectors, as bias_pass's row block
#pragma unroll
    for (int v = 0; v < 8; v++) {
      aie::store_v(dst + 32 * v, BF(aie::mac(A32(aie::load_v<32>(src + 32 * v)), aie::load_v<32>(bias + 32 * v), ones)));
      aie::store_v(dst + dstride + 32 * v, BF(aie::mac(A32(aie::load_v<32>(src + 256 + 32 * v)), aie::load_v<32>(bias + 32 * v), ones)));
    }
    src += 512;
    dst += 2 * dstride;
  }
}
// -> rb_bf2 (in place)
static inline void bias_rows(bfloat16 *c_out, const bfloat16 *bias) {
  bias_pcs((const float *)c_out, c_out, bias, 12, 512);           // pair 0, rb < 12 (C slots)
  bias_pcs(acc_hi, (bfloat16 *)acc_hi, bias, 6, 512);             // pair 0, rb >= 12
  bias_pcs(acc_hi + 256 * 6, (bfloat16 *)(acc_hi + 256 * 6), bias + 256, 18, 512);  // pair 1
}
// -> the drained layout: row block rb at C byte 1024 rb, half h at + 512 h
static inline void bias_to_c(bfloat16 *c_out, const bfloat16 *bias) {
#if defined(CSPLIT) && defined(EPLR)
  // round 19: rb 0..11 (both halves) first: they are the C object's part 1 (bytes 0 .. 12,287), released here so its drain overlaps
  // rb 12..17; the same values op for op (each pass reads only what no earlier pass wrote over)
  bias_pcs((const float *)c_out, c_out, bias, 12, 512);
  bias_pcs(acc_hi + 256 * 6, c_out + 256, bias + 256, 12, 512);
  release(c_out, 48 + CS_CONS, 1);
  bias_pcs(acc_hi, c_out + 512 * 12, bias, 6, 512);
  bias_pcs(acc_hi + 256 * 18, c_out + 256 + 512 * 12, bias + 256, 6, 512);
#else
  bias_pcs((const float *)c_out, c_out, bias, 12, 512);
  bias_pcs(acc_hi, c_out + 512 * 12, bias, 6, 512);
  bias_pcs(acc_hi + 256 * 6, c_out + 256, bias + 256, 18, 512);
#endif
}
#else
// every row block -> rb_bf (compact bf16, 8 blocks of 64 at the start of its fp32 area)
static inline void bias_rows(bfloat16 *c_out, const bfloat16 *bias) {
  bias_pass((const float *)c_out, c_out, bias, ACC_LO / 8, 1024);
  bias_pass(acc_hi, (bfloat16 *)acc_hi, bias, (m - ACC_LO) / 8, 1024);
}
#endif

// GELU by table (round 6): gelu(x) = x s(x) with s piecewise linear in 64 segments of 1/8 on [-4, 4) (x clamped
// into the table: s = ~0 / ~1 outside), looked up 16 lanes at a time (aie::lut<4, float, bfloat16> layout; the index
// is a byte offset, 16 B per segment). s goes hi/lo into the product. 3.9x faster than nk8::gelu_n on one core
// (16 vs 63 ns per 32 lanes, r6/kprobe.py) at the same rms error on real activations (r6/numh.py); the ViT MLP
// (gelu_pan, mlp0) uses the erf form torch's nn.GELU() computes, the others the tanh form.
constexpr int LUT_SHIFT = -SGELU_STEP + 4, LUT_BIAS_B = SGELU_BIAS * 16;
static_assert(SGELU_STEP == SERF_STEP && SGELU_BIAS == SERF_BIAS);
static inline __attribute__((always_inline)) B32 gelu_lut_i(B32 x, const float *ab, const float *cd) {
  const B32 xc = aie::max(aie::min(x, aie::broadcast<bfloat16, 32>(SGELU_HI - 1.0f / 64)), aie::broadcast<bfloat16, 32>(SGELU_LO));
  A32 sacc;
  B32 slope;
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const aie::vector<int32, 16> index =
        aie::add(aie::vector<int32, 16>(bfloat16_to_int(xc.template extract<16>(h), LUT_SHIFT)), LUT_BIAS_B);
    v32bfloat16 coeff0, coeff1;
    load_lut_2x_float(ab, cd, index, coeff0, coeff1);
    sacc.insert(h, aie::accum<accfloat, 16>((v16accfloat)::shuffle(coeff0, coeff1, T32_16x2_hi)));
    slope.insert(h, aie::vector<bfloat16, 32>(::shuffle(coeff0, coeff1, T16_16x4_lo)).template extract<16>(1));
  }
  const A32 s = aie::mac(sacc, slope, xc);
  const B32 sh = BF(s);
  const B32 sl = BF(aie::mac(s, sh, (bfloat16)-1.0f));
  return BF(aie::mac(aie::mul(x, sh), x, sl));
}
// bf16(gelu(v)) in place for n 32-lane vectors (n even)
#ifndef GELU_II_MIN  // round 11: with 8 Peano silently did NOT pipeline the loop (round 10 benched that serial loop)
#define GELU_II_MIN 4
#endif
#ifdef GELU_II  // round 10: one vector per trip through separate restrict pointers and a forced initiation interval: Peano
// then software-pipelines the chain at 17 bundles per 32 lanes (33.5 before; r10/gx.cc). In place is safe: a trip loads
// its vector before storing it, and later trips' loads touch later vectors only.
// Round 11: in / out must be separate restrict PARAMETERS (gx.cc); restrict copies of one pointer still alias for Peano and
// the loop stays serial.
static __attribute__((noinline)) void gelu_lut_io(const bfloat16 *__restrict in, bfloat16 *__restrict out, int n, const float *ab, const float *cd) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(GELU_II_MIN)
#pragma clang loop pipeline_initiation_interval(16)
  for (int i = 0; i < n; i++) {
    aie::store_v(out, gelu_lut_i(aie::load_v<32>(in), ab, cd));
    in += 32;
    out += 32;
  }
}
static inline void gelu_lut_n(bfloat16 *v, int n, const float *ab, const float *cd) { gelu_lut_io(v, v, n, ab, cd); }
#else
static __attribute__((noinline)) void gelu_lut_n(bfloat16 *v, int n, const float *ab, const float *cd) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int i = 0; i < 32 * n; i += 64) {
    B32 a = gelu_lut_i(aie::load_v<32>(v + i), ab, cd), b = gelu_lut_i(aie::load_v<32>(v + i + 32), ab, cd);
    aie::store_v(v + i, a);
    aie::store_v(v + i + 32, b);
  }
}
#endif

#if defined(GELU_MS) && !defined(BGELU)  // round 11: the GELU loop as a hand modulo schedule (r11/gen_gelu.py, bit-identical to gelu_lut_i): one call
// walks nv vectors in runs of 8 at a 64 B stride with `jump` bytes after each run (64: contiguous; 576: the first halves of
// consecutive 1 KB pieces), in place
#include "build/gelu_ms.h"
extern "C" void gelu_ms(const bfloat16 *in, bfloat16 *out, int trips, const float *ab, const float *cd, int jump);
#define GELU_RUN(v, nv, erf, jump) \
  gelu_ms(v, v, (nv) / GELU_MS_U, (erf) ? serf_ab : sgelu_ab, (erf) ? serf_cd : sgelu_cd, jump)
static_assert(m == 144 && n == 64, "GELU_MS runs assume 18 row blocks of 8 x 64 and 1 KB pieces");
#endif
#ifdef GUP  // round 12: r12/gen_gup.py -> build/gup.s / .h (the schedule replayed from r12/gup.json)
#if !defined(GELU_MS) || !defined(MM_PIECES)
#error "GUP needs GELU_MS and MM_PIECES"
#endif
#include "build/gup.h"
extern "C" void gup(const bfloat16 *v, const float *u, bfp16ebs8 *dst, int trips, const bfloat16 *bias, int jump);
#endif
#ifdef BGELU  // round 12: r12/gen_bgelu.py -> build/bgelu.s / .h (schedule from r12/bgelu.json)
#if !defined(GELU_FLAT) || !defined(GUP)
#error "BGELU needs GELU_FLAT (the flat-edge tables) and GUP"
#endif
#include "build/bgelu.h"
extern "C" void bgelu(const float *acc, bfloat16 *out, int trips, const float *ab, const float *cd, const bfloat16 *bias);
// bf16(gelu(bf16(acc + bias))) for npc 1 KB pieces at p, in place at each piece's start (bias_pcs + GELU_RUN in one loop)
#define BGELU_RUN(p, npc, erf, b) \
  bgelu((const float *)(p), (bfloat16 *)(p), 8 * (npc) / BGELU_U, (erf) ? serf_ab : sgelu_ab, (erf) ? serf_cd : sgelu_cd, b)
static_assert((8 * 6) % BGELU_U == 0);
// the three segments of a tile: pair 0 in the C slots (in place), pair 0 in acc_hi pieces 0-5 (-> o1), pair 1 (-> o2,
// skipped when o2 is null). One helper for yx / gated / mlp0: three call sites instead of eight (GEMM text, round 12)
static __attribute__((noinline)) void bgelu3(bfloat16 *c_out, const bfloat16 *bias, bool erf, bfloat16 *o1, bfloat16 *o2) {
  const float *ab = erf ? serf_ab : sgelu_ab, *cd = erf ? serf_cd : sgelu_cd;
  bgelu((const float *)c_out, c_out, 8 * 12 / BGELU_U, ab, cd, bias);
  bgelu(acc_hi, o1, 8 * 6 / BGELU_U, ab, cd, bias);
  if (o2) bgelu(acc_hi + 256 * 6, o2, 8 * 18 / BGELU_U, ab, cd, bias + 256);
}
#endif
#ifdef PAN  // round 12: r12/gen_pan.py -> build/pan.s / .h (schedule from r12/pan.json)
#include "build/pan.h"
extern "C" void pan(const bfloat16 *v, const bfloat16 *u, bfp16ebs8 *dst, int trips, int jump, int group_m1);
static_assert(12 % PAN_U == 0 && 6 % PAN_U == 0);
#endif
#ifdef VTK  // round 14: r14/gen_vt.py -> build/vtk.s / .h
#include "build/vtk.h"
extern "C" void vtk(const float *src, const bfloat16 *bias, bfp16ebs8 *dst, int32_t trips);
static_assert(VTK_U == 1);
#endif
#ifdef CPY32  // round 12: r12/gen_copy.py -> build/cpy32.s / .h
#include "build/cpy32.h"
extern "C" void cpy32(const void *src, void *dst, int trips, int src_jump, int dst_jump, int row_m1);
static_assert((18 * 9) % CPY32_U == 0);
#endif

#if defined(NO_GELU)  // timing differential only: the epilogues without the GELU
#define GELU_N(v, nv, erf) ((void)0)
#elif defined(GELU_NK8)  // round 5's arithmetic GELU (tanh form everywhere)
#define GELU_N(v, nv, erf) nk8::gelu_n(v, nv, kc)
#else
#define GELU_N(v, nv, erf) gelu_lut_n(v, nv, (erf) ? serf_ab : sgelu_ab, (erf) ? serf_cd : sgelu_cd)
#endif

extern "C" {

// fp32 accumulator + bias -> the epilogue selected by the object's flag. `c_out` is the C object.
static void acc_epi(bfloat16 *c_out, bfloat16 *bias) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int flag = ((const int *)bias)[256];
  const ExpC kc;
  if (flag <= 1) {  // plain / gelu, in place: block i's 64 fp32 at byte 256 i -> 64 bf16 at byte 128 i
#if defined(MM_PIECES) && defined(BGELU)
#elif defined(MM_PIECES)
    bias_to_c(c_out, bias);
#else
    bias_pass((const float *)c_out, c_out, bias, ACC_LO / 8, 512);
    bias_pass(acc_hi, c_out + ACC_LO * n, bias, (m - ACC_LO) / 8, 512);
#endif
#if defined(BGELU)  // round 12: yx's GELU tiles: bias + GELU fused into the drained layout (bias_to_c's order: pair 0 first,
    if (flag == 1)  // then pair 1 onto the second halves of the C slots, whose fp32 pair 0 has consumed)
      bgelu3(c_out, bias, false, c_out + 512 * 12, c_out + 256);
    else
      bias_to_c(c_out, bias);
    return;
#elif defined(GELU_MS)
    if (flag == 1) GELU_RUN(c_out, m * n / 32, false, 64);
#else
    if (flag == 1) GELU_N(c_out, m * n / 32, false);
#endif
    return;
  }
  if (flag == 7) {  // V^T objects: per object (48 keys) 8 dim groups of 6 transposed panels, via a temp at the end
    bfp16ebs8 *g_stash = (bfp16ebs8 *)c_out + VTMP_OFF;
#ifdef VTK  // round 14 (research T2): bias, transpose and push of 4 dim groups x 6 key blocks per call (r14/gen_vt.py)
#pragma clang loop unroll(disable)
    for (int obj = 0; obj < 3; obj++) {
      vtk(piece(c_out, obj * 6, 0), bias, g_stash, 24);
      vtk(piece(c_out, obj * 6, 1), bias + 256, g_stash + 4 * 448, 24);
      static_assert(CPY32_U == 3 && 3 * 38 * 32 >= 3584 && 3 * 38 * 32 <= 4608);  // (3,648 B: 64 B into the object's junk tail)
      cpy32(g_stash, c_out + obj * 4608 / 2, 38, 32, 32, 3 * 38 - 1);
    }
#else
    const B32 ones = aie::broadcast<bfloat16, 32>(1.0f);
    for (int obj = 0; obj < 3; obj++) {
      for (int j = 0; j < 8; j++) {
        B64 tv[6];
        for (int cb = 0; cb < 6; cb++) {
          const float *src = acc_cb(c_out, obj * 6 + cb, j);
          A32 a0(aie::load_v<32>(src)), a1(aie::load_v<32>(src + 32));
          a0 = aie::mac(a0, aie::load_v<32>(bias + 64 * j), ones);
          a1 = aie::mac(a1, aie::load_v<32>(bias + 64 * j + 32), ones);
          tv[cb] = aie::transpose(aie::concat(BF(a0), BF(a1)), 8, 8);
        }
        OutS ps(g_stash + j * 448);
        for (int cb = 0; cb < 6; cb++) ps.push(A64(tv[cb]).template to_vector<bfp16ebs8>());
      }
      bfloat16 *dst = c_out + obj * 4608 / 2;
#if defined(CPY32) && defined(VTCPY)  // round 14: the r12 copy loop; 3 x 38 units of 32 B = 3,648 B (64 B into the object's junk tail)
      static_assert(CPY32_U == 3 && 3 * 38 * 32 >= 3584 && 3 * 38 * 32 <= 4608);
      cpy32(g_stash, dst, 38, 32, 32, 3 * 38 - 1);
#else
      for (int i = 0; i < 3584 / 64; i++) aie::store_v(dst + 32 * i, aie::load_v<32>((const bfloat16 *)g_stash + 32 * i));
#endif
    }
#endif  // VTK
    return;
  }
#ifdef GUP  // round 12 (npu-research G1a): the up half + bias, gate x up, bf16 and the bfp push in one asm loop per segment
  if (flag == 2 || flag == 3) {  // (r12/gen_gup.py, bit-identical to the path below); the gate half keeps bias_pcs + GELU_MS
    // GUP_NO{BIAS,GELU,MS,STASH}: timing differentials only (wrong outputs)
#if defined(BGELU)  // round 12: the gate half's bias + GELU in one loop per segment
    bgelu3(c_out, bias, false, (bfloat16 *)acc_hi, nullptr);  // pair 0 only, in place
#else
#ifndef GUP_NOBIAS
    bias_pcs((const float *)c_out, c_out, bias, 12, 512);  // pair 0, rb < 12 (C slots)
    bias_pcs(acc_hi, (bfloat16 *)acc_hi, bias, 6, 512);    // pair 0, rb >= 12
#endif
#ifndef GUP_NOGELU
    GELU_RUN(c_out, 8 * 12, false, 576);
    GELU_RUN((bfloat16 *)acc_hi, 8 * 6, false, 576);
#endif
#endif
    uint8_t *dst = flag == 2 ? (uint8_t *)c_out + STASH_OFF : (uint8_t *)c_out;
    const int stride = flag == 2 ? STASH_STRIDE : 8 * PB;  // 4 blocks (288 B) per row block, then the 2D flush adds the rest
#ifndef GUP_NOMS
    gup(c_out, acc_hi + 256 * 6, (bfp16ebs8 *)dst, 48 / GUP_U, bias + 256, stride - 4 * PB);
    gup((const bfloat16 *)acc_hi, acc_hi + 256 * 18, (bfp16ebs8 *)(dst + 12 * stride), 24 / GUP_U, bias + 256, stride - 4 * PB);
#endif
#if defined(CPY32) && !defined(GUP_NOSTASH)  // round 12: the stash copy as a modulo-scheduled 32 B copy (r12/gen_copy.py)
    if (flag == 3)  // 18 rows of 288 B: stash slots (320 B) -> behind the own panels (576 B rows); 9 units per row
      cpy32((const uint8_t *)c_out + STASH_OFF, dst + 4 * PB, 18 * 9 / CPY32_U, STASH_STRIDE - 8 * 32, stride - 8 * 32, 8);
    if (false) {
#elif defined(GUP_NOSTASH)
    if (false) {
#else
    if (flag == 3) {
#endif  // [own 4 | stash 4] = 576 B at byte 576 rb: the stash copy behind each row block's own panels
      const uint8_t *g_stash = (const uint8_t *)c_out + STASH_OFF;
#pragma clang loop unroll(disable)
      for (int rb = 0; rb < m / 8; rb++) {
        const bfloat16 *s = (const bfloat16 *)(g_stash + rb * STASH_STRIDE);
        bfloat16 *d = (bfloat16 *)(dst + rb * stride + 4 * PB);
        for (int i = 0; i < 4 * PB / 16; i++) aie::store_v(d + 8 * i, aie::load_v<8>(s + 8 * i));
      }
    }
    return;
  }
#endif
#ifdef BGELU  // round 12: mlp0 (erf): bias + GELU fused over its three segments; flags 5 / 6 keep bias_rows
  if (flag == 4) {
    bgelu3(c_out, bias, true, (bfloat16 *)acc_hi, (bfloat16 *)(acc_hi + 256 * 6));  // both pairs in place
  } else
#endif
  bias_rows(c_out, bias);
#ifndef GUP
  if (flag == 2 || flag == 3) {  // gated: product panels of columns [0, 32) x [32, 64)
#if defined(GELU_MS) && defined(MM_PIECES)  // pair 0: rb < 12 in the C slots, rb >= 12 in acc_hi pieces 0-5
    GELU_RUN(c_out, 8 * 12, false, 576);
    GELU_RUN((bfloat16 *)acc_hi, 8 * 6, false, 576);
#else
#pragma clang loop unroll(disable)
    for (int rb = 0; rb < m / 8; rb++) GELU_N(rb_bf2(c_out, rb, 0), 8, false);
#endif
    // block-vector output streams write whole 64 B units: 4 panels (288 B) go to a 64 B-aligned stash slot
    // (stride 320) or straight to the object's k-step chunk; the stash is copied behind them in 16 B vectors
    bfp16ebs8 *g_stash = (bfp16ebs8 *)c_out + STASH_OFF;
    for (int rb = 0; rb < m / 8; rb++) {
      const bfloat16 *v = rb_bf2(c_out, rb, 0), *u = rb_bf2(c_out, rb, 1);
      bfp16ebs8 *dst = flag == 2 ? g_stash + rb * STASH_STRIDE : (bfp16ebs8 *)(c_out + rb * (8 * PB) / 2);
      B64 zv[4];  // compute first, then push (an output stream's state does not survive anything in between)
      for (int cb = 0; cb < 4; cb++)
        zv[cb] = aie::concat(BF(aie::mul(aie::load_v<32>(v + 64 * cb), aie::load_v<32>(u + 64 * cb))),
                             BF(aie::mul(aie::load_v<32>(v + 64 * cb + 32), aie::load_v<32>(u + 64 * cb + 32))));
      {
        OutS ps(dst);
        for (int cb = 0; cb < 4; cb++) ps.push(A64(zv[cb]).template to_vector<bfp16ebs8>());
      }
      if (flag == 3) {  // [own 4 | stash 4] = 576 B at byte 576 rb (always behind the reads of row block rb)
        const bfloat16 *s = (const bfloat16 *)(g_stash + rb * STASH_STRIDE);
        bfloat16 *d = (bfloat16 *)dst + 4 * PB / 2;
        for (int i = 0; i < 4 * PB / 16; i++) aie::store_v(d + 8 * i, aie::load_v<8>(s + 8 * i));
      }
    }
    return;
  }
#endif
  // 4/5/6: gelu -> panels (k-step chunk) / Q, K objects (3 x 3456 B at 4608 B): 8 panels per row block. Row block 0's
  // panels overwrite its own bf16 [0, 576) while it is read: panel cb ends at or before byte 128 (cb + 1), where
  // block cb + 1 starts, and every block is loaded before its panel is pushed.
#if defined(GELU_MS) && defined(MM_PIECES)  // both pairs: C slots (pair 0, rb < 12) and acc_hi pieces 0-23
#ifndef BGELU
  if (flag == 4) {
    GELU_RUN(c_out, 8 * 12, true, 576);
    GELU_RUN((bfloat16 *)acc_hi, 8 * 24, true, 576);
  }
#endif
  if (false) {
#else
  if (flag == 4) {
#endif
#pragma clang loop unroll(disable)
    for (int rb = 0; rb < m / 8; rb++) {  // the ViT MLP: erf form
#ifdef MM_PIECES
      GELU_N(rb_bf2(c_out, rb, 0), 8, true);
      GELU_N(rb_bf2(c_out, rb, 1), 8, true);
#else
      GELU_N(rb_bf2(c_out, rb, 0), 16, true);
#endif
    }
  }
#ifdef PAN  // round 12: the panel pushes as one modulo-scheduled asm loop per segment (r12/gen_pan.py, bit-identical)
  {  // flag 4: one contiguous k-step chunk; 5 / 6: 3 objects of 6 row blocks (3,456 B) at 4,608 B
    const int jump = flag == 4 ? 0 : 4608 - 6 * 8 * PB;
    pan(c_out, (const bfloat16 *)(acc_hi + 256 * 6), (bfp16ebs8 *)c_out, 12 / PAN_U, jump, 5);
    pan((const bfloat16 *)acc_hi, (const bfloat16 *)(acc_hi + 256 * 18),
        (bfp16ebs8 *)((uint8_t *)c_out + (flag == 4 ? 12 * 8 * PB : 2 * 4608)), 6 / PAN_U, jump, 5);
    return;
  }
#endif
  int obj = 0, orb = 0;  // flags 5/6: rb = 6 obj + orb (no division: __udivsi3 is a call)
#ifdef PAN_NOPUSH  // round 12 timing differential only (wrong outputs): no panel pushes
  if (false)
#endif
  for (int rb = 0; rb < m / 8; rb++) {
    const bfloat16 *v = rb_bf2(c_out, rb, 0), *u = rb_bf2(c_out, rb, 1);
    bfloat16 *dst = flag == 4 ? c_out + rb * (8 * PB) / 2 : c_out + (obj * 4608 + orb * 8 * PB) / 2;
    if (++orb == 6) { orb = 0; obj++; }
    OutS ps((bfp16ebs8 *)dst);
    for (int cb = 0; cb < 4; cb++) ps.push(A64(aie::load_v<64>(v + 64 * cb)).template to_vector<bfp16ebs8>());
    for (int cb = 0; cb < 4; cb++) ps.push(A64(aie::load_v<64>(u + 64 * cb)).template to_vector<bfp16ebs8>());
  }
}
void acc_epilogue(bfloat16 *c_out, bfloat16 *bias) {
#ifdef CSPLIT
  cs_take(c_out);
#endif
#ifdef PROF
  const uint32_t te = stamp();
#endif
#ifdef TPROF
  const uint32_t tte = tst();
#endif
  acc_epi(c_out, bias);
#if defined(CSPLIT) && defined(EPLR)
#if !defined(BGELU) || !defined(MM_PIECES)
#error "EPLR: only the BGELU / MM_PIECES epilogue releases the plain tiles' part 1 in bias_to_c"
#endif
  {  // the C cons lock's first release per object (part 1); IRON's out_c.release is the second (part 2). Plain tiles (flag 0) released
    const int flag = ((const int *)bias)[256];  // in bias_to_c; a gated object's first tile (flag 2) not at all
    if (flag != 0 && flag != 2) release(c_out, 48 + CS_CONS, 1);
  }
#endif
#ifdef TPROF  // record over the tile's first output words (timing build): magic 'TPRF'
  {
    const uint32_t t3 = tst();
    uint32_t *w = (uint32_t *)c_out;
    w[0] = 0x46525054u;
    w[1] = g_t1 - g_t0;  // first k-step incl. the B / A waits (k = 0 .. the k == 1 stamp)
    w[2] = g_t2 - g_t1;  // the other k-steps
    w[3] = g_tnks;
    w[4] = tte - g_t2;  // mm_ktile return -> epilogue entry
    w[5] = t3 - tte;  // epilogue
    w[6] = g_t0 - g_tlast;  // previous epilogue end -> this tile's mm_ktile entry (C acquire, loop code, ...)
    w[7] = g_t0;
    g_tlast = tst();
  }
#endif
#if defined(P35) && defined(KSC)
  g_bnext = (const bfp16ebs8 *)((uintptr_t)bias ^ ((uintptr_t)KS_B0 ^ (uintptr_t)KS_B1));
#endif
#ifdef KSP38  // timing build: over the tile's first output words (magic 'KS38')
  {
    uint32_t *w = (uint32_t *)c_out;
    w[0] = 0x3833534Bu;
    for (int i = 0; i < 5; i++) {
      w[1 + i] = g_kc[i];
      g_kc[i] = 0;
    }
  }
#endif
#ifdef PROF  // record: magic, cycles in mm_acc, calls, cycles acc_zero -> epilogue (k loop incl. waits), epilogue cycles
  const uint32_t t1 = stamp();
  uint32_t *w = (uint32_t *)c_out;
  w[0] = 0x464F5250u;
  w[1] = g_p_busy;
  w[2] = g_p_calls;
  w[3] = te - g_p_t0;
  w[4] = t1 - te;
  w[5] = g_p_gkb;
  w[6] = g_p_ga;
  w[7] = g_p_first;
  w[8] = g_p_gmax;
  w[9] = te - g_p_last;  // last call -> epilogue
  w[10] = g_p_hand;
  w[11] = g_p_zero;
  g_p_epi_end = stamp();
#endif
}
}

// ---- attention on the GEMM cores (attention v15's online-softmax kernel) --------------------------------------
// Per (pass, head): Q objects ([rb 6][8 dim panels], 3456 B) copied from the A ring, K objects (same layout) and
// V^T objects (8 dim groups of 6 transposed panels, 448 B each) from the B fifo, O accumulated in fp32 over the
// 24 key chunks, output = bf16(O / sum) as bfp panels at the C object's start (tile t -> row blocks 6t..6t+5).
// All working memory is aliased onto the C object (18,432 B) and acc_hi (20,480 B), which the GEMM does not use
// during the attention op.
// round 14 (research T3b): with -DKFOLD the K / V phase bodies are inlined into their entry points (no wrapper call frames)
#ifdef KFOLD
#define WRAP_FOLD inline __attribute__((always_inline))
#define WRAP_ATTR
#else
#define WRAP_FOLD __attribute__((noinline))
#define WRAP_ATTR __attribute__((minsize))
#endif
#if defined(RESCK) && !defined(ATTKB)
#error "RESCK passes srow through the score ring: only k_rb's order (test, rescale, exact rerun) allows it"
#endif
#ifdef RESCK
extern "C" void resc(float *o, float *osum, const bfloat16 *srow);
#endif
namespace at {
constexpr int QT = 48, HD = 64, KC = 48, NB = HD / 8, RB = QT / 8, CB = KC / 8;
constexpr int NKEYS = ATTN_KEYS;
constexpr int NCH = 24;
// The scores stay unscaled (q.k = SU x the softmax argument; exact since SU is a power of 2): the 1/SU lives in the exp
// table's slopes and index shift, in TAU / NEG and in the rescale argument (round 7: one mul + conversion less per
// score block, bit-identical)
constexpr int SU_LOG2 = 3;
constexpr float SU = 1 << SU_LOG2;
static_assert(ATTN_SCALE * SU == 1.0f);
constexpr float NEG = -64.0f * SU;
constexpr float TAU = 8.0f * SU;  // lazy max threshold (natural log units x SU)
constexpr int GQ = 576, GV = 448, OBJQ = 3456;
static_assert(NKEYS % 16 == 0 && NKEYS <= NCH * KC);
using MM = aie::mmul<8, 8, 8, bfp16ebs8>;
using BP = aie::block_vector<bfp16ebs8, 64>;
template <aie_dm_resource R>
using InS = aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, 64, R>;
using B16 = aie::vector<bfloat16, 16>;
static float *g_o0;            // O tile 0 (the C object; tile 1 is acc_hi's start): [rb 6][dim block 8][64 fp32]
static float *g_osum;          // row sums: [tile][rb][64 fp32]
static bfloat16 *g_srb;        // scores of one row block [cb][64]
static bfloat16 *g_maxb;       // running max [tile][rb][16]
static bfloat16 *g_mbuf;       // [64]
static bfp16ebs8 *g_ones;      // V^T group whose dim 0 is all ones (6 panels)
static int g_ntiles, g_chunk;
static float *g_lcd;          // the exp table's second copy (C object tail, another bank than aexp_ab)

// acc_hi's part of the map is static (round 14: addresses, not pointers set at run time: -112 B of program memory).
// acc_hi (24,576 B): O tile 1 [0, 12288), the Q copies (3456 B each, from 12288), the P buffers (2688 B each, from 19200)
static inline float *o_rows(int t, int rb) { return (t ? acc_hi : g_o0) + rb * NB * 64; }
static inline bfp16ebs8 *q_buf(int t) { return (bfp16ebs8 *)((uint8_t *)acc_hi + 12288 + 3456 * t); }
static inline bfp16ebs8 *erb(int t) { return (bfp16ebs8 *)((uint8_t *)acc_hi + 19200 + 2688 * t); }
static_assert(19200 + 2 * 2688 == (m - ACC_LO) * n * 4);
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

// P = exp(s - m) by table (round 7, exp_lut.h from r7/explut.py): segment i = floor(4 d) + bias with d = bf16(s - m)
// clamped into the table (the 4 bottom segments are zero: P = 0 below e^-10 of the stored max); then
// p = off_i + slope_i (s - m) = (off_i - slope_i m) + slope_i s in fp32 (s, m are bf16: exact products, no hi/lo
// split of d), converted to bfp16 straight from the accumulator (round 6 rounded P to bf16 first). Real-score
// emulation (r7/expemu.py, 5 blocks): the softmax weights' row-norm error is 0.91-0.96x round 6's expd_i.
constexpr int AEXP_SHIFT = -AEXP_STEP + 4 - SU_LOG2, AEXP_BIAS_B = AEXP_BIAS * 16;
static_assert(AEXP_SLOPE_SCALE * SU == 1.0f);
// stage 1: the clamped table argument
static inline __attribute__((always_inline)) B32 aexp_arg(B32 s, A32 negm) {
  const B32 dh = BF(aie::mac(negm, s, (bfloat16)1.0f));
  return aie::max(aie::min(dh, aie::broadcast<bfloat16, 32>(8.9375f * SU)), aie::broadcast<bfloat16, 32>(AEXP_LO * SU));
}
// stage 2: lookup (16 lanes per load_lut) and the line
static inline __attribute__((always_inline)) A32 aexp_val(B32 dc, B32 s, B32 mv, const float *ab, const float *cd) {
  A32 off;
  B32 slope;
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const aie::vector<int32, 16> index =
        aie::add(aie::vector<int32, 16>(bfloat16_to_int(dc.template extract<16>(h), AEXP_SHIFT)), AEXP_BIAS_B);
    v32bfloat16 coeff0, coeff1;
    load_lut_2x_float(ab, cd, index, coeff0, coeff1);
    off.insert(h, aie::accum<accfloat, 16>((v16accfloat)::shuffle(coeff0, coeff1, T32_16x2_hi)));
    slope.insert(h, aie::vector<bfloat16, 32>(::shuffle(coeff0, coeff1, T16_16x4_lo)).template extract<16>(1));
  }
  return aie::mac(aie::msc(off, slope, mv), slope, s);
}
// the row block's 6 P panels (one 448 B group) from its scores; invalid key blocks hold -1024 SU (-> the zero segments).
// Two stages by hand: block cb + 1's argument is computed while block cb looks up and pushes (Peano does not overlap
// the chain across trips; the last trip's look-ahead reads the 128 B behind g_srb, unused). 49 bundles per 64 scores
// incl. the push (round 6: 65 for the exp + a separate push pass); carrying the looked-up lines instead: 65 (spills).
static __attribute__((noinline)) void aexp_push(const bfloat16 *v, bfp16ebs8 *dst, B64 mrow, const float *ab, const float *cd) {
  const B32 m0 = mrow.template extract<32>(0), m1 = mrow.template extract<32>(1);
  const A32 nm0 = aie::mul(m0, (bfloat16)-1.0f), nm1 = aie::mul(m1, (bfloat16)-1.0f);
  OutS es(dst);
  B32 c0 = aexp_arg(aie::load_v<32>(v), nm0), c1 = aexp_arg(aie::load_v<32>(v + 32), nm1);
  AIE_PREPARE_FOR_PIPELINING
  for (int cb = 0; cb < CB; cb++) {
    const bfloat16 *q = v + cb * 64;
#ifdef ATTN_NOEXP  // timing differential only
    es.push(A64(aie::load_v<64>(q)).template to_vector<bfp16ebs8>());
#else
    const B32 n0 = aexp_arg(aie::load_v<32>(q + 64), nm0), n1 = aexp_arg(aie::load_v<32>(q + 96), nm1);
    const A32 p0 = aexp_val(c0, aie::load_v<32>(q), m0, ab, cd), p1 = aexp_val(c1, aie::load_v<32>(q + 32), m1, ab, cd);
    es.push(aie::concat(p0, p1).template to_vector<bfp16ebs8>());
    c0 = n0;
    c1 = n1;
#endif
  }
}
static inline void exp_rb(int t, int rb, B64 mrow) {  // invalid blocks: -1024 SU (chunk_k)
  aexp_push(g_srb, erb(t) + rb * GV, mrow, aexp_ab, g_lcd);
#ifdef ATTN_ETWICE  // timing twin only: the exp + push twice
  aexp_push(g_srb, erb(t) + rb * GV, mrow, aexp_ab, g_lcd);
#endif
}

static __attribute__((noinline, minsize)) void rescale_rb(int t, int rb, B16 dm) {
  ExpC k;
  B32 d = aie::concat(dm, aie::zeros<bfloat16, 16>());
  // rows8 of the 8 scales (dm's lanes 8-15 repeat 0-7): a transpose of 4 copies (round 14: -48 B of text, same values)
#ifdef RESCK  // (exp32 inline: rescale_rb was its only caller; -16 B of program memory)
  const B16 e = nk8::exp32_i(d, k, aie::zeros<bfloat16, 32>()).template extract<16>(0);
#else
  const B16 e = nk8::exp32(d, k).template extract<16>(0);
#endif
  const B64 srow = aie::transpose(aie::concat(e, e, e, e), 8, 8);
#ifdef RESCK  // round 15 (research rev 10 S1): the O / row-sum update in asm (r15/gen_resc.py, bit-identical); srow passes through
  // the score ring's first slot (the exact test has read the scores; the exact rerun rewrites the ring)
  aie::store_v(g_srb, srow);
  resc(o_rows(t, rb), g_osum + (t * RB + rb) * 64, g_srb);
}
#else
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
#endif

// scores -> running max / rescale -> P (bfp panels, one row block at a time; P kept per tile for the V phase)
static __attribute__((noinline)) void chunk_k(int t, const bfp16ebs8 *q, const bfp16ebs8 *kt, int nkb) {
  for (int rb = 0; rb < RB; rb++) {
    // scores of all CB key blocks (a constant trip count; the invalid blocks of the last chunk are masked after)
    for (int cb = 0; cb < CB; cb += 2) {
      MM acc0, acc1, acc2, acc3;
      {  // Q popped per key block, on the other load slot (round 6 held Q in 8 registers: spills, ~70 bundles per pair;
         // now 53), 2 streams live; the same MAC order as round 6 (bit-identical)
#pragma unroll
        for (int u = 0; u < 2; u++) {
          InS<aie_dm_resource::a> qs(q);
          InS<aie_dm_resource::b> ks(kt);
          qs.seek_byte(rb * GQ);
          ks.seek_byte((cb + u) * GQ);
          MM &ae = u ? acc1 : acc0, &ao = u ? acc3 : acc2;
#pragma unroll
          for (int j = 0; j < NB; j += 2) {
            BP qa = qs.pop(), ka = ks.pop(), qb = qs.pop(), kb = ks.pop();
            if (j == 0) { ae.mul(qa, aie::op_transpose(ka)); ao.mul(qb, aie::op_transpose(kb)); }
            else { ae.mac(qa, aie::op_transpose(ka)); ao.mac(qb, aie::op_transpose(kb)); }
          }
        }
      }
      A64 t0 = aie::add(acc0.to_accum(), acc2.to_accum()), t1 = aie::add(acc1.to_accum(), acc3.to_accum());
      aie::store_v(g_srb + cb * 64, BF(t0));
      aie::store_v(g_srb + (cb + 1) * 64, BF(t1));
    }
    for (int cb = nkb; cb < CB; cb++) aie::store_v(g_srb + cb * 64, aie::broadcast<bfloat16, 64>(-1024.0f * SU));
    B64 mx = aie::max(aie::load_v<64>(g_srb), aie::load_v<64>(g_srb + 64));
#pragma unroll
    for (int cb = 2; cb < CB; cb++) mx = aie::max(mx, aie::load_v<64>(g_srb + cb * 64));
    bfloat16 *mb = g_maxb + (t * RB + rb) * 16;
    B16 m_old = aie::load_v<16>(mb);
    B16 m_new = aie::max(rowmax8(mx), m_old);
#ifndef ATTN_NORESCALE  // timing differential only
    // lazy max: exp(s - m) only has to stay in range, so the stored max moves (and O is rescaled) only when a row's
    // max passed it by more than TAU (P <= e^TAU; O and the sums are fp32, P is bfp16 with per-block exponents).
    // Rescaling on every increase cost 0.42 ms per block.
    if (!aie::lt(aie::add(m_old, aie::broadcast<bfloat16, 16>(TAU)), m_new).empty()) {
      aie::store_v(mb, m_new);
      rescale_rb(t, rb, BF(aie::mul(BF(aie::mac(aie::accum<accfloat, 16>(m_old), m_new, (bfloat16)-1.0f)), (bfloat16)(1.0f / SU))));
    }
#endif
    exp_rb(t, rb, rows8(mb));
    // the P panels of this row block go to the tile's P buffer slot rb (6 x 448 B per tile)
  }
}

#ifdef ATTK_BENCH  // r13/attkb.cc timing differentials (outputs wrong then): bit 0 skips the asm QK, bit 1 the glue / test, bit 2 aexpk,
int g_attk_mode = 0;  // bit 3 forces the fused driver's slow path
#define ATTK_SKIP(b) (g_attk_mode & (b))
#elif defined(ATTK_FORCE)  // the same bits fixed at compile time: in-situ timing builds (outputs wrong)
#define ATTK_SKIP(b) ((ATTK_FORCE) & (b))
#else
#define ATTK_SKIP(b) 0
#endif
#if defined(ATTK) || defined(ATTK_BOTH)  // round 13 (npu-research P48 / P49a): QK^T and the exp + push as generated asm loops
// (r13/gen_attqk.py, r13/gen_aexp.py; bit-identical to chunk_k above). The max / lazy rescale / rows8 glue stays C++ here.
}  // namespace at
extern "C" void attqk(const bfp16ebs8 *q, const bfp16ebs8 *kt, bfloat16 *srb, int32_t trips);
extern "C" void aexpk(const bfloat16 *s, bfp16ebs8 *dst, const float *ab, const float *cd, const bfloat16 *mrow, int32_t trips);
#include "build/attqk.h"
#include "build/aexp.h"
namespace at {
static_assert(CB % ATTQK_U == 0 && CB % AEXPK_U == 0);
static __attribute__((noinline)) void chunk_k_asm(int t, const bfp16ebs8 *q, const bfp16ebs8 *kt, int nkb) {
  for (int rb = 0; rb < RB; rb++) {
    if (!ATTK_SKIP(1)) attqk((const bfp16ebs8 *)((const uint8_t *)q + rb * GQ), kt, g_srb, CB / ATTQK_U);
    bfloat16 *mb = g_maxb + (t * RB + rb) * 16;
    if (!ATTK_SKIP(2)) {
      for (int cb = nkb; cb < CB; cb++) aie::store_v(g_srb + cb * 64, aie::broadcast<bfloat16, 64>(-1024.0f * SU));
      B64 mx = aie::max(aie::load_v<64>(g_srb), aie::load_v<64>(g_srb + 64));
#pragma unroll
      for (int cb = 2; cb < CB; cb++) mx = aie::max(mx, aie::load_v<64>(g_srb + cb * 64));
      B16 m_old = aie::load_v<16>(mb);
      B16 m_new = aie::max(rowmax8(mx), m_old);
#ifndef ATTN_NORESCALE
      if (!aie::lt(aie::add(m_old, aie::broadcast<bfloat16, 16>(TAU)), m_new).empty()) {
        aie::store_v(mb, m_new);
        rescale_rb(t, rb, BF(aie::mul(BF(aie::mac(aie::accum<accfloat, 16>(m_old), m_new, (bfloat16)-1.0f)), (bfloat16)(1.0f / SU))));
      }
#endif
      aie::store_v(g_mbuf, rows8(mb));
    }
#ifdef ATTK_CPPEXP  // bench: the asm QK^T with today's C++ exp
    if (!ATTK_SKIP(4)) aexp_push(g_srb, erb(t) + rb * GV, aie::load_v<64>(g_mbuf), aexp_ab, g_lcd);
#else
    if (!ATTK_SKIP(4)) aexpk(g_srb, (bfp16ebs8 *)((uint8_t *)erb(t) + rb * GV), aexp_ab, g_lcd, g_mbuf, CB / AEXPK_U);
#endif
  }
}
#endif
#if defined(ATTKF) || defined(ATTKF_BOTH)  // round 13 (npu-research P56, per row block): QK^T + exp(s - stored max) + push in ONE
// asm loop (r13/gen_attk.py); the exact lazy-max test runs after it on the stored scores, and the row blocks whose max moves
// (~4 % after chunk 0) plus chunks with invalid key blocks take today's slow path over their P slot: bit-identical.
}  // namespace at
extern "C" int32_t attk(const bfp16ebs8 *q, const bfp16ebs8 *kt, bfp16ebs8 *dst, const bfloat16 *mrow, bfloat16 *srb, const float *cd,
                        int32_t trips);  // returns the row block's flag: some lane had s - m_old >= TAU - 0.25 SU
#include "build/attk.h"
namespace at {
static_assert(CB % ATTK_U == 0);
#ifndef ATTK_SKIP
#define ATTK_SKIP(b) 0
#endif
#ifdef ATTK_COUNT
static uint32_t g_attk_slow = 0;
#endif
// today's exact path over the stored scores of one row block: the max test, the lazy rescale, P with the stored max
static __attribute__((noinline, minsize)) void slow_rb(int t, int rb, bfp16ebs8 *dst, int nkb) {
  bfloat16 *mb = g_maxb + (t * RB + rb) * 16;
  const B16 m_old = aie::load_v<16>(mb);
#pragma clang loop unroll(disable)
  for (int cb = nkb; cb < CB; cb++) aie::store_v(g_srb + cb * 64, aie::broadcast<bfloat16, 64>(-1024.0f * SU));
  B64 mx = aie::max(aie::load_v<64>(g_srb), aie::load_v<64>(g_srb + 64));
#pragma clang loop unroll(disable)
  for (int cb = 2; cb < CB; cb++) mx = aie::max(mx, aie::load_v<64>(g_srb + cb * 64));
  const B16 m_new = aie::max(rowmax8(mx), m_old);
  if (!aie::lt(aie::add(m_old, aie::broadcast<bfloat16, 16>(TAU)), m_new).empty()) {  // today's test: the max moves
    aie::store_v(mb, m_new);
#ifdef ATTKE  // round 14: chunk 0 has O and the row sums at +0 (attn_begin), so its rescale is a no-op (+0 x finite = +0)
    if (g_chunk != 0)
#endif
      rescale_rb(t, rb, BF(aie::mul(BF(aie::mac(aie::accum<accfloat, 16>(m_old), m_new, (bfloat16)-1.0f)), (bfloat16)(1.0f / SU))));
  }
#ifdef ATTK_COUNT  // debug build: count the slow units (attn_end writes the count over the output's first words; the unit
  g_attk_slow++;  // count is known: 4 heads x 3 tiles x 6 row blocks x 23 chunks per block per core)
#endif
  if (!ATTK_SKIP(16)) aexp_push(g_srb, dst, rows8(mb), aexp_ab, g_lcd);
}
#ifdef ATTKB  // round 14 (research P63): r14/gen_attkb.py, the fused loop over all row blocks of a (tile, chunk) in one call
}  // namespace at
#ifdef ATTKS2MAX  // round 17 (research rev 11 S2): the loop keeps a running max per row block in `slots` and returns its flags' OR
extern "C" int32_t attkb(const bfp16ebs8 *q, const bfp16ebs8 *kt, bfp16ebs8 *dst, const bfloat16 *mrows, bfloat16 *srb, bfloat16 *slots,
                         int32_t trips, int32_t hi);
#else
extern "C" void attkb(const bfp16ebs8 *q, const bfp16ebs8 *kt, bfp16ebs8 *dst, const bfloat16 *mrows, bfloat16 *srb, int32_t *flags,
                      int32_t trips, int32_t hi);  // hi: bf16 bits of the clamp / flag bound
#endif
#include "build/attkb.h"
#if defined(ATTKS2MAX) != defined(ATTKB_S2)
#error "build/attkb.h does not match -DATTKS2MAX (r14/gen_attkb.py --s2)"
#endif
namespace at {
static_assert(ATTKB_U == 1 && 896 == CB * 128 + 128, "attkb finds the exp table copy at srb + 896");
// One row block: the fast pass over its nkb key blocks (clamp and flag bound 63.75 SU); if it flagged (or `force`: chunk 0, the
// batched call's flagged row blocks) today's exact path: the max test over the scores in the ring, the lazy rescale, and P
// again by the same loop with the (possibly new) stored max and aexp_push's clamp 71.5 SU: op for op today's exp (round 14,
// research T3a: aexp_push's 464 B gone). The last chunk's invalid key blocks: P = exp(-1024 SU) = all-zero bfp blocks.
constexpr int32_t HI_FAST = 0x427F, HI_EXACT = 0x428F;  // bf16 63.75 / 71.5 (= 8.9375 SU, aexp_arg's bound)
#ifdef KRB_STAMP  // bench only (r13/attkb.cc): core-timer stamps of k_rb's phases (first pass, exact test, rescale, rerun)
uint32_t g_krb[8];
#define KRB_T(i) (g_krb[i] = read_tm(0x340F8), __builtin_aie2p_sched_barrier())
#else
#define KRB_T(i) ((void)0)
#endif
#ifdef ATTKS2MAX
// One row block with its running-max slot `sl` (64 bf16): unless `force` (a batched call has filled the slot and flagged this row
// block), the fast pass first (the slot initialised to -inf); then, if flagged, today's exact test on the slot's maximum (the same
// scores and max operand order as the ring-based test: bit-identical), the lazy rescale and the exact rerun (clamp 71.5 SU).
static __attribute__((noinline, minsize)) void k_rb(int t, int rb, const bfp16ebs8 *q, const bfp16ebs8 *kt, int nkb, bfloat16 *sl, bool force) {
  bfp16ebs8 *dst = (bfp16ebs8 *)((uint8_t *)erb(t) + rb * GV);
  const bfp16ebs8 *qr = (const bfp16ebs8 *)((const uint8_t *)q + rb * GQ);
  bfloat16 *mr = (bfloat16 *)((uint8_t *)kt + 3456) + 64 * rb;
  KRB_T(0);
  int32_t f = 1;
  if (!force) {
    aie::store_v(sl, aie::broadcast<bfloat16, 64>(-__builtin_inff()));
    if (!ATTK_SKIP(1)) f = attkb(qr, kt, dst, mr, g_srb, sl, nkb, HI_FAST);
  }
  KRB_T(1);
  if (ATTK_SKIP(2)) return;
  if (f || ATTK_SKIP(8) || ATTK_SKIP(32)) {
    bfloat16 *mb = g_maxb + (t * RB + rb) * 16;
    const B16 m_old = aie::load_v<16>(mb);
    B64 mx = aie::load_v<64>(sl);
    if (nkb < CB) mx = aie::max(mx, aie::broadcast<bfloat16, 64>(-1024.0f * SU));  // today's invalid key blocks
    const B16 m_new = aie::max(rowmax8(mx), m_old);
    KRB_T(2);
    if (!aie::lt(aie::add(m_old, aie::broadcast<bfloat16, 16>(TAU)), m_new).empty()) {  // today's test: the max moves
      aie::store_v(mb, m_new);
      if (g_chunk != 0)  // chunk 0: O and the row sums are +0 (attn_begin), the rescale a no-op
        rescale_rb(t, rb, BF(aie::mul(BF(aie::mac(aie::accum<accfloat, 16>(m_old), m_new, (bfloat16)-1.0f)), (bfloat16)(1.0f / SU))));
      const B16 nm = aie::neg(m_new);
      aie::store_v(mr, aie::transpose(aie::concat(nm, nm, nm, nm), 8, 8));
    }
    KRB_T(3);
#ifdef ATTK_COUNT
    g_attk_slow++;
#endif
    if (!ATTK_SKIP(16)) attkb(qr, kt, dst, mr, g_srb, sl, nkb, HI_EXACT);
    KRB_T(4);
  }
  constexpr int NKB_LAST = (NKEYS % KC) / 8;  // nkb < CB only in the last chunk
  static_assert(NKB_LAST > 0 && PB * NKB_LAST % 16 == 0);
  static_assert(PB * NKB_LAST % 32 == 16 && PB * CB % 32 == 16, "one 16 B store at each end, 32 B stores between");
  if (nkb < CB) {  // straight stores: a loop costs ~15 bundles per trip under minsize
    uint8_t *z = (uint8_t *)dst + PB * NKB_LAST;
    aie::store_v((int32_t *)z, aie::zeros<int32, 4>());
#pragma unroll
    for (int i = 16; i < PB * (CB - NKB_LAST) - 16; i += 32) aie::store_v((int32_t *)(z + i), aie::zeros<int32, 8>());
    aie::store_v((int32_t *)(z + PB * (CB - NKB_LAST) - 16), aie::zeros<int32, 4>());
  }
}
static WRAP_FOLD void chunk_k_fused(int t, const bfp16ebs8 *q, const bfp16ebs8 *kt, int nkb) {
  // rows8 of -m_old per row block in the K object's junk tail (as below) and the row blocks' max slots at -inf
  bfloat16 *mr = (bfloat16 *)((uint8_t *)kt + 3456);
  const bfloat16 *mx = g_maxb + t * RB * 16;
  alignas(64) bfloat16 sl[RB * 64];
#pragma clang loop unroll(disable)
  for (int rb = 0; rb < RB; rb++) {
    const B16 nm = aie::neg(aie::load_v<16>(mx + 16 * rb));
    aie::store_v(mr + 64 * rb, aie::transpose(aie::concat(nm, nm, nm, nm), 8, 8));
    aie::store_v(sl + 64 * rb, aie::broadcast<bfloat16, 64>(-__builtin_inff()));
  }
  if (nkb == CB && g_chunk != 0) {  // chunk 0 (every row block flags) and the last chunk go row block by row block
    int32_t f = 1;
    if (!ATTK_SKIP(1)) f = attkb(q, kt, erb(t), mr, g_srb, sl, RB * CB, HI_FAST);
    if ((!f && !ATTK_SKIP(8) && !ATTK_SKIP(32)) || ATTK_SKIP(128)) return;  // the common case: nothing flagged
#pragma clang loop unroll(disable)
    for (int rb = 0; rb < RB; rb++) {
      // the row block's flag from its maximum: bf16(max s + (-m_old)) > the bound in some lane, the loop's own test per score
      // (s + (-m) in fp32 then bf16 is monotone in s)
      const B64 d = BF(aie::add(A64(aie::load_v<64>(sl + 64 * rb)), A64(aie::load_v<64>(mr + 64 * rb))));
      if (!aie::lt(aie::broadcast<bfloat16, 64>(63.75f), d).empty() || ATTK_SKIP(8) || ATTK_SKIP(32))
        k_rb(t, rb, q, kt, CB, sl + 64 * rb, true);
    }
    return;
  }
#pragma clang loop unroll(disable)
  for (int rb = 0; rb < RB; rb++) k_rb(t, rb, q, kt, nkb, sl + 64 * rb, false);
}
#else
static __attribute__((noinline, minsize)) void k_rb(int t, int rb, const bfp16ebs8 *q, const bfp16ebs8 *kt, int nkb, int32_t *fl, bool force) {
  bfp16ebs8 *dst = (bfp16ebs8 *)((uint8_t *)erb(t) + rb * GV);
  const bfp16ebs8 *qr = (const bfp16ebs8 *)((const uint8_t *)q + rb * GQ);
  bfloat16 *mr = (bfloat16 *)((uint8_t *)kt + 3456) + 64 * rb;
  KRB_T(0);
  if (!ATTK_SKIP(1)) attkb(qr, kt, dst, mr, g_srb, fl, nkb, HI_FAST);
  KRB_T(1);
  if (ATTK_SKIP(2)) return;
  if (force || fl[nkb] || ATTK_SKIP(8) || ATTK_SKIP(32)) {
    bfloat16 *mb = g_maxb + (t * RB + rb) * 16;
    const B16 m_old = aie::load_v<16>(mb);
#pragma clang loop unroll(disable)
    for (int cb = nkb; cb < CB; cb++) aie::store_v(g_srb + cb * 64, aie::broadcast<bfloat16, 64>(-1024.0f * SU));
    B64 mx = aie::load_v<64>(g_srb);
#pragma clang loop unroll(disable)
    for (int cb = 1; cb < CB; cb++) mx = aie::max(mx, aie::load_v<64>(g_srb + cb * 64));
    const B16 m_new = aie::max(rowmax8(mx), m_old);
    KRB_T(2);
    if (!aie::lt(aie::add(m_old, aie::broadcast<bfloat16, 16>(TAU)), m_new).empty()) {  // today's test: the max moves
      aie::store_v(mb, m_new);
      if (g_chunk != 0)  // chunk 0: O and the row sums are +0 (attn_begin), the rescale a no-op
        rescale_rb(t, rb, BF(aie::mul(BF(aie::mac(aie::accum<accfloat, 16>(m_old), m_new, (bfloat16)-1.0f)), (bfloat16)(1.0f / SU))));
      const B16 nm = aie::neg(m_new);
      aie::store_v(mr, aie::transpose(aie::concat(nm, nm, nm, nm), 8, 8));
    }
    KRB_T(3);
#ifdef ATTK_COUNT
    g_attk_slow++;
#endif
    if (!ATTK_SKIP(16)) attkb(qr, kt, dst, mr, g_srb, fl, nkb, HI_EXACT);
    KRB_T(4);
  }
  constexpr int NKB_LAST = (NKEYS % KC) / 8;  // nkb < CB only in the last chunk
  static_assert(NKB_LAST > 0 && PB * NKB_LAST % 16 == 0);
  static_assert(PB * NKB_LAST % 32 == 16 && PB * CB % 32 == 16, "one 16 B store at each end, 32 B stores between");
  if (nkb < CB) {  // straight stores: a loop costs ~15 bundles per trip under minsize
    uint8_t *z = (uint8_t *)dst + PB * NKB_LAST;
    aie::store_v((int32_t *)z, aie::zeros<int32, 4>());
#pragma unroll
    for (int i = 16; i < PB * (CB - NKB_LAST) - 16; i += 32) aie::store_v((int32_t *)(z + i), aie::zeros<int32, 8>());
    aie::store_v((int32_t *)(z + PB * (CB - NKB_LAST) - 16), aie::zeros<int32, 4>());
  }
}
static WRAP_FOLD void chunk_k_fused(int t, const bfp16ebs8 *q, const bfp16ebs8 *kt, int nkb) {
  // rows8 of -m_old per row block, in the K object's junk tail (bytes 3456..4224 of the 4608 B object; this core holds it).
  // g_maxb's 16 lanes hold the 8 row maxima twice, so 4 copies form an [8 copies][8 rows] block and its transpose is rows8
  bfloat16 *mr = (bfloat16 *)((uint8_t *)kt + 3456);
  const bfloat16 *mx = g_maxb + t * RB * 16;
#pragma clang loop unroll(disable)
  for (int rb = 0; rb < RB; rb++) {
    const B16 nm = aie::neg(aie::load_v<16>(mx + 16 * rb));
    aie::store_v(mr + 64 * rb, aie::transpose(aie::concat(nm, nm, nm, nm), 8, 8));
  }
  // one flag per (row block, key block), then their OR; behind them k_rb's own (its reruns must not overwrite the batched flags)
  alignas(32) int32_t fl[RB * CB + 1 + CB + 1];
  int32_t *fl1 = fl + RB * CB + 1;
  if (nkb == CB && g_chunk != 0) {  // chunk 0 (every row block flags) and the last chunk go row block by row block
    if (!ATTK_SKIP(1)) attkb(q, kt, erb(t), mr, g_srb, fl, RB * CB, HI_FAST);
    if ((!fl[RB * CB] && !ATTK_SKIP(8) && !ATTK_SKIP(32)) || ATTK_SKIP(128)) return;  // the common case: nothing flagged
    // (timing twin bit 128: the batched chunks' flags ignored; chunk 0 / the last chunk keep the exact path)
#pragma clang loop unroll(disable)
    for (int rb = 0; rb < RB; rb++) {
      int32_t f = 0;
#pragma clang loop unroll(disable)
      for (int k = 0; k < CB; k++) f |= fl[CB * rb + k];
      if (f || ATTK_SKIP(8) || ATTK_SKIP(32)) k_rb(t, rb, q, kt, CB, fl1, true);  // the rerun puts this row block's scores in the ring
    }
    return;
  }
#pragma clang loop unroll(disable)
  for (int rb = 0; rb < RB; rb++) k_rb(t, rb, q, kt, nkb, fl, false);
}
#endif  // ATTKS2MAX
#else
static __attribute__((noinline)) void chunk_k_fused(int t, const bfp16ebs8 *q, const bfp16ebs8 *kt, int nkb) {
  for (int rb = 0; rb < RB; rb++) {
    bfp16ebs8 *dst = (bfp16ebs8 *)((uint8_t *)erb(t) + rb * GV);
    {  // rows8 of -m_old for the loop (the arg's fp32 add and the line's vmac both take -m: the same products and roundings)
      alignas(32) bfloat16 nm[16];
      aie::store_v(nm, aie::neg(aie::load_v<16>(g_maxb + (t * RB + rb) * 16)));
      aie::store_v(g_mbuf, rows8(nm));
    }
    // scores -> g_srb (+ one garbage block behind the last valid one), P with m_old -> dst
    int32_t flag = 1;
#ifdef ATTKE  // round 14: only the valid key blocks (the last chunk has nkb = 2); an unflagged row block's invalid panels are
    // today's exp of -1024 SU = 0 (the zero segments), i.e. all-zero bfp blocks: stored directly, no slow path
    constexpr int NKB_LAST = (NKEYS % KC) / 8;  // the one chunk with invalid key blocks (2 valid of 6 at 1072 keys)
    static_assert(ATTK_U == 1 && NKB_LAST > 0 && PB * NKB_LAST % 16 == 0);
    if (!ATTK_SKIP(1)) flag = attk((const bfp16ebs8 *)((const uint8_t *)q + rb * GQ), kt, dst, g_mbuf, g_srb, g_lcd, nkb);
    if (ATTK_SKIP(2)) continue;
    if (flag || ATTK_SKIP(8) || ATTK_SKIP(32)) slow_rb(t, rb, dst, nkb);
    else if (nkb < CB) {
#pragma unroll
      for (int z = PB * NKB_LAST; z < PB * CB; z += 16) aie::store_v((int32_t *)((uint8_t *)dst + z), aie::zeros<int32, 4>());
    }
#else
    if (!ATTK_SKIP(1)) flag = attk((const bfp16ebs8 *)((const uint8_t *)q + rb * GQ), kt, dst, g_mbuf, g_srb, g_lcd, CB / ATTK_U + (ATTK_SKIP(64) ? 1 : 0));
    if (ATTK_SKIP(2)) continue;
    // unflagged (no lane within TAU - 0.25 of the stored max): the max cannot have moved, the fast P is today's. Flagged
    // (chunk 0 always: m_old = NEG), or invalid key blocks in the chunk: today's exact test and path over the stored scores
    if (flag || nkb < CB || ATTK_SKIP(8) || ATTK_SKIP(32)) slow_rb(t, rb, dst, nkb);  // bench bit 32: ignore the flag
#endif
  }
}
#endif  // ATTKB
#endif

#ifdef ATTV  // round 14 (research P50): O += P V as one generated asm loop per (tile, chunk) (r14/gen_attv.py), the row sums in C++
}  // namespace at
#ifdef V2T  // round 18: both tiles per call (r14/gen_attv.py --two): jump = 64 + tile 1's O rows - (tile 0's + 12,288 B)
extern "C" void attv(const bfp16ebs8 *p, const bfp16ebs8 *v, float *o, int32_t trips, int32_t jump);
#else
extern "C" void attv(const bfp16ebs8 *p, const bfp16ebs8 *v, float *o, int32_t trips);
#endif
extern "C" void attvs(const bfp16ebs8 *p, const bfp16ebs8 *ones, float *osum, int32_t trips);  // the row sums, same loop form
#include "build/attv.h"
#include "build/attvs.h"
namespace at {
static_assert(ATTV_U == 1 && NB == 8 && RB % ATTVS_U == 0);
#ifdef V2T
// every tile of the pass in one attv and one attvs call: the P buffers and the row sums are contiguous across tiles, O jumps
static_assert(19200 + 2688 == 19200 + RB * GV && RB * NB * 64 * 4 == 12288);
static WRAP_FOLD void chunk_v2(int nt, const bfp16ebs8 *vc) {
  attv(erb(0), vc, o_rows(0, 0), nt * RB * NB / 2, (int32_t)((uint8_t *)o_rows(1, 0) - ((uint8_t *)o_rows(0, 0) + 12288)) + 64);
  attvs(erb(0), g_ones, g_osum, nt * RB / ATTVS_U);
}
#endif
static WRAP_FOLD void chunk_v(int t, const bfp16ebs8 *vc) {
#ifdef V2T
  attv(erb(t), vc, o_rows(t, 0), RB * NB / 2, 0);
#else
  attv(erb(t), vc, o_rows(t, 0), RB * NB / 2);
#endif
  attvs(erb(t), g_ones, g_osum + t * RB * 64, RB / ATTVS_U);
}
#else
// O += P [V | 1] for the 6 row blocks of tile t
static WRAP_FOLD void chunk_v(int t, const bfp16ebs8 *vc) {
  for (int rb = 0; rb < RB; rb++) {
    float *orow = o_rows(t, rb);
    BP preg[CB];
    {
      InS<aie_dm_resource::a> ps(erb(t) + rb * GV);
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
#endif  // ATTV
}  // namespace at

extern "C" {
// c_out: the C object of the first head of the pass (only used for its address; every head's object is at the
// same L1 address since the C sub-fifo is single-buffered)
__attribute__((minsize)) void attn_begin(bfloat16 *c_out, int32_t ntiles) {  // minsize (program memory); 4 stores per loop trip
#ifdef ZHI
  g_hi_clean = 0;
#endif
#ifdef CSPLIT
  cs_take(c_out);
#endif
  using namespace at;
#ifdef ATTN_STUB
  return;
#endif
#ifdef PROF
  const uint32_t tb0 = stamp();
  ga_pre = tb0 - ga_last;
  ga_k = ga_v = ga_gap = 0;
#endif
  aie::set_rounding(aie::rounding_mode::conv_even);
  uint8_t *cb = (uint8_t *)c_out;
  // acc_hi (24,576 B, static: its address is known before the C object is acquired): O tile 1, the Q copies, the
  // P buffers. The C object (18,432 B): O tile 0, the row sums, scores, maxima, the ones panels, the exp scratch.
  // (round 3 gave g_srb 384 B: the 6 score blocks are 6 x 128 B = 768 B, so the running maxima aliased the scores
  // of key blocks 3-5 and every row block's softmax saw 16 garbage P entries: corr 0.91 / 0.69 instead of 1.0)
  // Round 13: the K loop's hot buffers first. The C object spans L1 banks 2 and 3 (35072 .. 53504) and the K objects (B fifo)
  // are in bank 3: with the scores, the max rows and the exp table's second copy at the object's start (bank 2) the loop's
  // K pops are bank 3's only traffic (two accesses to one bank in a bundle stall the core; r8/mmb.py). attn_end's output
  // panels overwrite this start last, in the order the O rows are consumed (as before, when O tile 0 was here).
  g_srb = (bfloat16 *)cb;               // 768 B = CB x 64 bf16 (to 768)
  g_mbuf = (bfloat16 *)(cb + 768);      // 128 B (to 896): rows8 of the stored max; the fused K loop's one garbage score
                                        // block lands here (dead by then), so it must follow g_srb
  g_lcd = (float *)(cb + 896);          // 1280 B (to 2176): the exp table's second copy (bank 2; aexp_ab: bank 0)
  g_o0 = (float *)(cb + 2176);          // 12288 B (to 14464)
  g_osum = (float *)(cb + 14464);       // 3072 B (to 17536)
  g_maxb = (bfloat16 *)(cb + 17536);    // 384 B (to 17920)
  g_ones = (bfp16ebs8 *)(cb + 17920);   // 432 B, 64 B-aligned (to 18352 <= 18432)
  static_assert(CB * 64 * 2 == 768 && 896 + AEXP_N * 16 == 2176 && 17920 + 432 <= m * n * 2);
  for (int i = 0; i < AEXP_N * 4 / 16; i++) aie::store_v(g_lcd + 16 * i, aie::load_v<16>(aexp_ab + 16 * i));
  g_ntiles = ntiles;
  g_chunk = 0;
#ifdef ZASM  // round 14 (text): r14/zero2.s zeroes 64 B at a + 128 B at b per trip, so a region of 192 t bytes is zero2(R, R + 64 t, t):
  // O tile 0 + both tiles' row sums are contiguous (2176 .. 17536 of the C object: 80 x 192 B), O tile 1 is acc_hi's first 64 x 192 B
  static_assert(RB * NB * 64 * 4 + 2 * RB * 64 * 4 == 80 * 192 && RB * NB * 64 * 4 == 64 * 192);
  zero2(g_o0, g_o0 + 80 * 16, 80);  // (g_osum = g_o0 + RB * NB * 64: attn_begin's layout)
  if (ntiles > 1) zero2(acc_hi, acc_hi + 64 * 16, 64);
#else
  for (int t = 0; t < ntiles; t++)  // only the tiles of this pass
    for (float *o = o_rows(t, 0), *e = o + RB * NB * 64; o < e; o += 128) {
      aie::store_v(o, aie::zeros<float, 32>());
      aie::store_v(o + 32, aie::zeros<float, 32>());
      aie::store_v(o + 64, aie::zeros<float, 32>());
      aie::store_v(o + 96, aie::zeros<float, 32>());
    }
  for (int i = 0; i < 2 * RB * 64 / 32; i++) aie::store_v(g_osum + 32 * i, aie::zeros<float, 32>());
#endif
  for (int i = 0; i < 2 * RB * 16 / 32; i++) aie::store_v(g_maxb + 32 * i, aie::broadcast<bfloat16, 32>(NEG));
  // the ones panels: 1.0 in every row (dim) of each 8x8 panel, so every lane of a row's sum holds the row sum
  // (attn_end and rescale_rb work in the row layout). The bfp conversion of 1.0 = mantissa 64 at exponent 127 (round
  // 6 wrote the 432 bytes one at a time: ~7,300 cycles per head group).
  OutS os(g_ones);
  for (int cb = 0; cb < CB; cb++) os.push(A64(aie::broadcast<bfloat16, 64>(1.0f)).template to_vector<bfp16ebs8>());
#ifdef PROF
  ga_last = stamp();
  ga_beg = ga_last - tb0;
#endif
}

// the A ring's 3 sub-tiles hold Q object (head hh of the current triple, tile t): copy it if it is this core's
__attribute__((minsize)) void attn_qcopy(bfp16ebs8 *a0, bfp16ebs8 *a1, bfp16ebs8 *a2, int32_t hh, int32_t t, int32_t pass, int32_t myrow) {
  using namespace at;
#ifdef ATTN_STUB
  return;
#endif
  // the Q copies arrive before this head's C object is acquired: only acc_hi may be written here. Pass 0 keeps tiles 0 and 1,
  // pass 1 tile 2 (round 14: the test and the copy in their smallest form: -48 B of program memory)
  if (hh != myrow || (pass == 0) != (t < 2)) return;
  aie::vector<int32_t, 16> *d = (aie::vector<int32_t, 16> *)q_buf(t - 2 * pass);
  bfp16ebs8 *const s[3] = {a0, a1, a2};
#pragma clang loop unroll(disable)
  for (int i = 0; i < 3; i++) {
    const aie::vector<int32_t, 16> *p = (const aie::vector<int32_t, 16> *)s[i];
#pragma clang loop unroll(disable)
    for (int w = 0; w < 1152 / 64; w++) *d++ = p[w];
  }
}

WRAP_ATTR void attn_chunk_k(bfp16ebs8 *kobj) {
  using namespace at;
#if defined(ATTN_STUB) || defined(ATTN_STUB_K)
  return;
#endif
  event0();
#ifdef PROF
  const uint32_t tk0 = stamp();
  ga_gap += tk0 - ga_last;
#endif
  const int valid = NKEYS - g_chunk * KC;
  const int nkb = valid >= KC ? CB : (valid <= 0 ? 0 : valid / 8);
  if (nkb > 0) {
#if defined(ATTKF)
    for (int t = 0; t < g_ntiles; t++) chunk_k_fused(t, q_buf(t), kobj, nkb);
#elif defined(ATTK)
    for (int t = 0; t < g_ntiles; t++) chunk_k_asm(t, q_buf(t), kobj, nkb);
#else
    for (int t = 0; t < g_ntiles; t++) chunk_k(t, q_buf(t), kobj, nkb);
#endif
#ifdef ATTN_KTWICE  // timing twin only: the K phase twice (outputs as once: the max is already in place)
    for (int t = 0; t < g_ntiles; t++) chunk_k(t, q_buf(t), kobj, nkb);
#endif
  }
#ifdef PROF
  ga_last = stamp();
  ga_k += ga_last - tk0;
#endif
  event1();
}

#ifdef ATTV
__attribute__((minsize))  // round 14: with the V phase in asm this is two calls per tile: minsize costs nothing
#else
WRAP_ATTR
#endif
void attn_chunk_v(bfp16ebs8 *vobj) {
  using namespace at;
#if defined(ATTN_STUB) || defined(ATTN_STUB_V)
  return;
#endif
#ifdef PROF
  const uint32_t tv0 = stamp();
  ga_gap += tv0 - ga_last;
#endif
  const int valid = NKEYS - g_chunk * KC;
#if defined(ATTV) && defined(V2T)
  if (valid > 0) chunk_v2(g_ntiles, vobj);
#elif defined(ATTV)  // straight calls (round 14, text: Peano kept 11 callee-saved registers across the tile loop)
  if (valid > 0) {
    chunk_v(0, vobj);
    if (g_ntiles > 1) chunk_v(1, vobj);
  }
#else
  if (valid > 0)
    for (int t = 0; t < g_ntiles; t++) chunk_v(t, vobj);
#endif
#ifdef ATTN_VTWICE  // timing twin only: the V phase twice (O wrong)
  if (valid > 0)
    for (int t = 0; t < g_ntiles; t++) chunk_v(t, vobj);
#endif
  if (++g_chunk == NCH) g_chunk = 0;
#ifdef PROF
  ga_last = stamp();
  ga_v += ga_last - tv0;
#endif
}

// o = bf(O * bf(1/sum)) -> bfp panels at the C object's start: tile t, row block rb -> [6 t + rb][8 dim panels].
// Every lane of a row's sum holds the sum (the ones panels are ones in all 8 dims), so 1/sum is a 64-lane vector
// reciprocal in the row layout (round 5: aie::inv on bf16(sum), the scalar unit, then 8 lane extracts).
#if defined(ATTK_COUNT) || defined(END_MIN)
__attribute__((minsize))  // the counting build: program memory (timing irrelevant); round 14 END_MIN: text for P50
#endif
void attn_end(bfloat16 *c_out) {  // -O2 (round 7)
  using namespace at;
#if defined(ATTN_STUB) || defined(ATTN_STUB_END)
  return;
#endif
#ifdef PROF
  const uint32_t te0 = stamp();
  ga_gap += te0 - ga_last;
#endif
  const ExpC kc;
  for (int t = 0; t < g_ntiles; t++) {
    for (int rb = 0; rb < RB; rb++) {
      const float *os = g_osum + (t * RB + rb) * 64;
      A32 s0(aie::load_v<32>(os)), s1(aie::load_v<32>(os + 32));
      B64 rrow = aie::concat(BF(nk8::recipa_i(s0, kc)), BF(nk8::recipa_i(s1, kc)));
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
#ifdef ATTK_COUNT  // over the first output words (debug build): magic 'ATTC', slow units, units (cumulative per core)
  {
    uint32_t *w = (uint32_t *)c_out;
    w[0] = 0x43545441u;
    w[1] = g_attk_slow;
  }
#endif
#ifdef PROF  // record over the first output words (timing build): magic 'ATTP', the head group's phase cycles
  ga_last = stamp();
  uint32_t *w = (uint32_t *)c_out;
  w[0] = 0x50545441u;
  w[1] = ga_pre;
  w[2] = ga_beg;
  w[3] = ga_k;
  w[4] = ga_v;
  w[5] = ga_gap;
  w[6] = ga_last - te0;
  w[7] = g_ntiles;
#endif
#ifdef CSPLIT
  g_zh = 0;
#ifdef EPLR
  release(c_out, 48 + CS_CONS, 1);  // part 1 (IRON's out_c.release is part 2)
#endif
#endif
}
}
