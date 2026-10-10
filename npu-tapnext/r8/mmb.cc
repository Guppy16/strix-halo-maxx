// GEMM-call bench (round 8): one core, no DMA, the asm mm_core (r7/gen_mm.py) called n times between two core-timer
// reads, its A / B / C operands at host-chosen offsets in a static pool. Finds in-call memory stalls (bank conflicts)
// without the rest of the step around it. Operand values are garbage (timing only).
#include <stdint.h>

#include <aie_api/aie.hpp>

extern "C" void mm_core(const void *pA, const void *pB, float *pC);
extern "C" void mm_core_split(const void *pA, const void *pB, float *pCload, float *pCstore);

constexpr int POOL = POOL_BYTES;
alignas(64) static uint8_t pool[POOL];

static inline __attribute__((always_inline)) uint32_t stamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}

// in: [0] n calls, [1] mode (0 mm_core, 1 split, 2 empty loop), [2] offA, [3] offB, [4] offC, [5] offC store (split)
// out: [0] cycles, [1] pool address, [2..] echo
extern "C" void mmb(int32_t *in, int32_t *out) {
  const int n = in[0], mode = in[1];
  const uint8_t *pA = pool + in[2], *pB = pool + in[3];
  float *pC = (float *)(pool + in[4]), *pS = (float *)(pool + in[5]);
  const uint32_t t0 = stamp();
  if (mode == 0) {
#pragma clang loop unroll(disable)
    for (int i = 0; i < n; i++) mm_core(pA, pB, pC);
  } else if (mode == 1) {
#pragma clang loop unroll(disable)
    for (int i = 0; i < n; i++) mm_core_split(pA, pB, pC, pS);
  }
  const uint32_t t1 = stamp();
  out[0] = (int32_t)(t1 - t0);
  out[1] = (int32_t)(uintptr_t)pool;
}
