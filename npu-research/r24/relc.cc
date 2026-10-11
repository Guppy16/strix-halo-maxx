// pass 19 RELC bench: does rel.cond release for any non-zero condition, or test one bit? A shim -> core DMA writes tagged 1,152 B
// objects into a0, gated by lock aF (id 4, init 0: nothing may land until the core frees the slot) and releasing aU (id 5).
// Per test value v: a0 gets a sentinel, the core calls relc(v) (rel.cond #52, r5, r26 = v), spins, and reads a0[0]: a tag means
// the conditional release happened (the DMA wrote), the sentinel that it did not. Then the core balances (an unconditional release
// if it did not happen), waits for the object (acquire aU) and reads its tag. Nothing can block forever.
#include <aie_api/aie.hpp>
#include <stdint.h>
extern "C" void relc(int32_t cond);
static void spin(int n) {
  for (volatile int i = 0; i < n; i++) {
  }
}
extern "C" void relcb(int32_t *x, int32_t *y, uint32_t *a0) {
  const int32_t vals[4] = {0, 2, 1, 3};
  for (int t = 0; t < 4; t++) {
    a0[0] = 0xDEAD0000u + t;
    relc(vals[t]);
    spin(20000);
    const uint32_t seen = a0[0];
    const int happened = seen != 0xDEAD0000u + t;
    if (!happened) release(a0, 48 + 4, 1);
    acquire_greater_equal(a0, 48 + 5, 1);  // the object landed (aU)
    y[3 * t] = vals[t];
    y[3 * t + 1] = happened;
    y[3 * t + 2] = (int32_t)a0[0];
  }
  // consume every object the shim streams (8), whatever the semantics turned out to be: no task is left pending at the end
  spin(20000);
  const int written = (int)(a0[0] & 0xFFFFu) + 1;
  for (int k = written; k < 8; k++) release(a0, 48 + 4, 1);
  for (int k = 4; k < 8; k++) acquire_greater_equal(a0, 48 + 5, 1);
  y[19] = written;
  y[20] = x[0];
}
