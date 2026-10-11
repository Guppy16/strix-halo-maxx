// One-core bench of the GEMM k-step with a DMA-FED A ring (npu-research pass 15, main's round-18 request). After r19/rotb.cc,
// whose A slots were statics pre-released nine at a time, so an acquire never blocked and round 18's g2f race (acq.cond one
// bundle before the A fill) passed the bench. Here the three A slots are IRON buffers filled by a shim -> core DMA with the
// deployed lock ids (aF = 4 free, aU = 5 full; the S2MM BD acquires aF and releases aU per 1,152 B object), so every
// acq.cond of the kernel waits for real DMA data and a too-early fill reads a slot the DMA has not finished.
// The A objects of a k-step are A_OBJ(p) = slot pattern p % 3 (the same bytes rotb.cc generated on the core); the host
// generates them with the same LCG (rotdb.py a_host), the core advances its LCG past them and generates B as rotb.cc does.
// Output: y[0] cycles of the last call, y[1] calls, y[2 + 2 rb + pair] piece hashes (must match between ks and rot, and with
// rotb's pre-released mode), y[38] record count.
#include "mm8.cc"
#include "mm8_rot.inc"

#ifndef KSB_SEED
#define KSB_SEED 12345u
#endif
alignas(64) static uint8_t ksb_b[4608];
alignas(64) static uint8_t ksb_br[4608] __attribute__((section(".aie.bank3")));
alignas(64) static bfloat16 ksb_c[DIM_M * DIM_N];

static inline __attribute__((always_inline)) uint32_t ksb_stamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}

extern "C" void rotdb(int32_t *x, int32_t *y, bfp16ebs8 *a0, bfp16ebs8 *a1, bfp16ebs8 *a2) {
  uint32_t s = KSB_SEED;
  auto rnd = [&]() { s = s * 1664525u + 1013904223u; return s >> 8; };
#pragma clang loop unroll(disable)
  for (int i = 0; i < 3 * 1152; i++) (void)rnd();  // the A bytes come from the DMA: only advance the generator past them
  for (int i = 0; i < 4608; i++) ksb_b[i] = (i % 9 == 0) ? (uint8_t)(120 + rnd() % 8) : (uint8_t)rnd();
#pragma clang loop unroll(disable)
  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 72; j++) ksb_br[72 * i + j] = ksb_b[72 * mm_rot_border[i] + j];
  const int n = ROTDB_N;  // every object the shim streams is consumed (a partial run would leave the DMA blocked on aF)
  acc_zero(ksb_c);
  uint32_t t0 = 0, t1 = 0;
  for (int c = 0; c < n; c++) {
    if (c == n - 1) t0 = ksb_stamp();
#if ROTB_ROT
    rot_kstep(a0, a1, a2, (bfp16ebs8 *)ksb_br, ksb_c);
#else
    mm_kstep(a0, a1, a2, (bfp16ebs8 *)ksb_b, ksb_c);
#endif
    if (c == n - 1) t1 = ksb_stamp();
  }
  for (int rb = 0; rb < DIM_M / 8; rb++)
    for (int pair = 0; pair < 2; pair++) {
      const uint32_t *w = (const uint32_t *)piece(ksb_c, rb, pair);
      uint32_t h = 2166136261u;
      for (int i = 0; i < 256; i++) h = (h ^ w[i]) * 16777619u;
      y[2 + 2 * rb + pair] = (int32_t)h;
    }
  y[0] = (int32_t)(t1 - t0);
  y[1] = n;
  y[38] = MM_ROT_NREC;
}
