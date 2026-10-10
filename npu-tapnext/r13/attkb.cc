// round 13: one-core bench of the attention K phase: mm8.cc's C++ chunk_k against chunk_k_asm (r13/gen_attqk.py +
// r13/gen_aexp.py), same Q / K / state, bit-exact compare of the P panels, the maxima and the scores, checksums of O and
// the row sums. Built by r11/apb.py build --cc r13/attkb.cc --asm <attqk.s + aexp.s>.
#define ATTK_BENCH
#ifdef ATTKF_BENCH
#define ATTKF_BOTH
#define CHUNK_K_NEW chunk_k_fused
#else
#define ATTK_BOTH
#define CHUNK_K_NEW chunk_k_asm
#endif
#include "../mm8.cc"

// the C object stand-in (attn_begin's working memory); round 14: used from tile address 0x78900 as in situ (C0_x_buff_0 at
// 35072: the scores, -m rows and the exp table copy in bank 2) when that window lies inside this array, else from its start
alignas(64) static uint8_t g_c[18432 + 8192];
static uint8_t *cobj() {
  const uintptr_t a = (uintptr_t)g_c, want = 0x78900;
  return (a <= want && want + 18432 <= a + sizeof(g_c)) ? (uint8_t *)want : g_c;
}
// round 14: the K object in bank 3 and Q in bank 1, as in situ (full12r14a: B objects at 53504 / 58112, the Q copies at acc_hi + 12288
// / + 15744 = bank 1; this bench's acc_hi starts at 0x70f00, so its q_buf(1) is the one wholly in bank 1)
alignas(64) static uint8_t g_k[4608 + 64] __attribute__((section(".aie.bank3")));  // the K object (+ a guard)
static uint32_t g_href[3];                     // reference hashes: P panels, maxima (6 x 16 bf16), scores

static inline __attribute__((always_inline)) uint32_t gstamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}

static uint32_t g_seed;
static float rnd(float scale) {  // LCG in [-scale, scale)
  g_seed = g_seed * 1664525u + 1013904223u;
  return ((int)(g_seed >> 9) - (1 << 22)) * (scale / (1 << 22));
}

static __attribute__((noinline, minsize)) void panels(bfp16ebs8 *dst, int n, float scale) {  // n bfp panels of 8 x 8 random bf16 values
  OutS os(dst);
  for (int p = 0; p < n; p++) {
    alignas(64) bfloat16 v[64];
    for (int i = 0; i < 64; i++) v[i] = (bfloat16)rnd(scale);
    os.push(A64(aie::load_v<64>(v)).template to_vector<bfp16ebs8>());
  }
}

static __attribute__((noinline, minsize)) void setup(int seed, float scale, int warm) {
  using namespace at;
  attn_begin((bfloat16 *)cobj(), 1);
  g_seed = 12345u + seed;
  panels(q_buf(1), 48, scale);
  panels((bfp16ebs8 *)g_k, 48, scale);
  for (int i = 3456; i < 4608 + 64; i++) g_k[i] = (uint8_t)(i * 7);  // the object's junk tail: anything
  if (warm) {  // O and the sums: random, so a rescale shows (round 14: cold = chunk 0, where attn_begin leaves them at +0)
    for (int i = 0; i < RB * NB * 64; i++) g_o0[i] = rnd(3.0f);
    for (int i = 0; i < RB * 64; i++) g_osum[i] = rnd(3.0f) + 10.0f;
  }
  g_chunk = warm ? 1 : 0;
  for (int i = 0; i < 2688 / 4; i++) ((uint32_t *)erb(0))[i] = 0xA5A5A5A5u;
  if (warm) {  // the stored maxima above every score: the lazy-rescale branch is not taken
    for (int i = 0; i < RB * 16; i++) g_maxb[i] = (bfloat16)(warm * 64.0f);
  }
}

static __attribute__((noinline, minsize)) uint32_t sum32(const void *p, int words) {
  uint32_t s = 0;
  for (int i = 0; i < words; i++) s = s * 31u + ((const uint32_t *)p)[i];
  return s;
}

// x: [1] seed, [2] scale x 0.25, [3] nkb, [4] warm (0: maxima at NEG; else maxima at warm * 64), [5] asm skip mask (timing only)
// y: [0] C++ cycles, [1] asm cycles, [2] P panels differ (hash), [3..5] unused, [6] maxima differ, [7] scores differ,
//    [8] O checksum diff (0 = same), [9] osum diff, [10] the K object's junk tail changed
extern "C" __attribute__((minsize)) void apb(int32_t *x, int32_t *y) {  // program memory: the bench code is cold
  using namespace at;
  aie::set_rounding(aie::rounding_mode::conv_even);
#ifdef KRB_STAMP
  if (x[0] == 2) {  // round 14: k_rb's phases on one flagged row block (x[1] = 1: the max moves and O is rescaled; 0: chunk 0)
    setup(0, 4.0f, 2);
    g_chunk = x[1];
    for (int i = 0; i < RB * 16; i++) g_maxb[i] = (bfloat16)(x[1] ? -64.0f : NEG);  // every row's max moves (scores ~ +-10)
#ifdef ATTKS2MAX  // round 17: the first pass fills the row block's max slot (force = false: it runs inside k_rb)
    alignas(64) bfloat16 sl[64];
    k_rb(0, 0, q_buf(1), (const bfp16ebs8 *)g_k, CB, sl, false);
#else
    alignas(32) int32_t fl[CB + 1];
    k_rb(0, 0, q_buf(1), (const bfp16ebs8 *)g_k, CB, fl, true);
#endif
    for (int i = 0; i < 4; i++) y[i] = (int32_t)(g_krb[i + 1] - g_krb[i]);
    return;
  }
#endif
  if (x[0] == 3) {  // round 14: the V phase on random P / V^T / O / sums: hashes (compare two builds) and cycles
    setup(x[1], 1.0f, 2);
    g_seed = 777u + x[1];
    for (int rb = 0; rb < RB; rb++) panels((bfp16ebs8 *)((uint8_t *)erb(0) + rb * GV), CB, 1.0f);
    for (int g = 0; g < NB; g++) panels((bfp16ebs8 *)(g_k + g * GV), CB, 2.0f);
    for (int i = 0; i < RB * NB * 64; i++) o_rows(0, 0)[i] = rnd(3.0f);
    const uint32_t t0 = gstamp();
    chunk_v(0, (const bfp16ebs8 *)g_k);
    const uint32_t t1 = gstamp();
    y[0] = (int32_t)(t1 - t0);
    y[1] = (int32_t)sum32(o_rows(0, 0), RB * NB * 64);
    y[2] = (int32_t)sum32(g_osum, RB * 64);
    for (int b = 0; b < RB * NB; b++) y[8 + b] = (int32_t)sum32(o_rows(0, 0) + 64 * b, 64);
    return;
  }
  if (x[0] == 1) {  // round 14: acc_zero over the C object stand-in and acc_hi (64 B guards behind both)
    uint8_t *c = cobj();
    for (int i = 0; i < 18432 + 64; i++) c[i] = (uint8_t)(i * 13 + 1);
    for (int i = 0; i < 24576 / 4; i++) ((uint32_t *)acc_hi)[i] = 0xDEADBEEFu + i;
    const uint32_t g0 = sum32(c + 12288, 64 / 4);
    const uint32_t t0 = gstamp();
    acc_zero((bfloat16 *)c);
    const uint32_t t1 = gstamp();
    int nz = 0, nh = 0;
    for (int i = 0; i < 12288 / 4; i++) nz += ((uint32_t *)c)[i] != 0;
    for (int i = 0; i < 24576 / 4; i++) nh += ((uint32_t *)acc_hi)[i] != 0;
    y[0] = (int32_t)(t1 - t0);
    y[1] = nz;
    y[2] = nh;
    y[3] = sum32(c + 12288, 64 / 4) != g0;
    return;
  }
  const int seed = x[1], nkb = x[3], warm = x[4];
  g_attk_mode = x[5];
  const float scale = x[2] * 0.25f;
  setup(seed, scale, warm);
  uint32_t t0 = gstamp();
  chunk_k(0, q_buf(1), (const bfp16ebs8 *)g_k, nkb);
  uint32_t t1 = gstamp();
  g_href[0] = sum32(erb(0), 2688 / 4);
  g_href[1] = sum32(g_maxb, 192 / 4);
  g_href[2] = sum32(g_srb, 768 / 4);
  if (x[6]) {  // dump: score block x[6] - 1 (64 bf16: C++ at y[16..47], asm at y[48..79]) and P panel 0 (C++ y[80..97], asm y[98..115])
    for (int i = 0; i < 32; i++) y[16 + i] = ((const int32_t *)(g_srb + 64 * (x[6] - 1)))[i];
    for (int i = 0; i < 18; i++) y[80 + i] = ((const int32_t *)((const uint8_t *)erb(0) + 72 * x[7]))[i];  // P panel x[7] of row block 0
  }
  for (int cb = 0; cb < 6 && !x[6]; cb++) y[16 + cb] = (int32_t)sum32(g_srb + 64 * cb, 32);  // per score block (the last row block's)
  for (int rb = 0; rb < 6 && !x[6]; rb++) y[32 + rb] = (int32_t)sum32((const uint8_t *)erb(0) + 448 * rb, 108);  // per row block's P
  for (int rb = 0; rb < 6 && !x[6]; rb++) y[48 + rb] = (int32_t)sum32(g_maxb + 16 * rb, 8);
  const uint32_t so = sum32(g_o0, RB * NB * 64), ss = sum32(g_osum, RB * 64);
  setup(seed, scale, warm);
  const uint32_t sj = sum32(g_k + 4608, 64 / 4);  // round 14: the guard only (ATTKB keeps the -m rows in the junk tail)
  uint32_t t2 = gstamp();
  CHUNK_K_NEW(0, q_buf(1), (const bfp16ebs8 *)g_k, nkb);
  uint32_t t3 = gstamp();
  const int bad = sum32(erb(0), 2688 / 4) != g_href[0], first = -1;
  if (x[6]) {
    for (int i = 0; i < 32; i++) y[48 + i] = ((const int32_t *)(g_srb + 64 * (x[6] - 1)))[i];
    for (int i = 0; i < 18; i++) y[98 + i] = ((const int32_t *)((const uint8_t *)erb(0) + 72 * x[7]))[i];
  }
  for (int cb = 0; cb < 6 && !x[6]; cb++) y[24 + cb] = (int32_t)sum32(g_srb + 64 * cb, 32);
  for (int rb = 0; rb < 6 && !x[6]; rb++) y[40 + rb] = (int32_t)sum32((const uint8_t *)erb(0) + 448 * rb, 108);
  for (int rb = 0; rb < 6 && !x[6]; rb++) y[56 + rb] = (int32_t)sum32(g_maxb + 16 * rb, 8);
  const int badm = sum32(g_maxb, 192 / 4) != g_href[1], bads = sum32(g_srb, 768 / 4) != g_href[2];
  y[0] = (int32_t)(t1 - t0);
  y[1] = (int32_t)(t3 - t2);
  y[2] = bad;
  y[3] = first;
  y[4] = 0;
  y[5] = 0;
  y[6] = badm;
  y[7] = bads;
  y[8] = sum32(g_o0, RB * NB * 64) != so;
  y[9] = sum32(g_osum, RB * 64) != ss;
  y[10] = sum32(g_k + 4608, 64 / 4) != sj;  // the junk tail + guard untouched (garbage pops are reads only)
}
