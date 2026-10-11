// round 18 DMAB (r18/dmab.py): segments of one 64 B access per cycle on a probe buffer while the memtile DMA writes the ring
#include <aie_api/aie.hpp>

extern "C" {
void dmab_ld(const void *p, int trips);
void dmab_st(void *p, int trips);
void dmab_nop(const void *p, int trips);
}

static inline __attribute__((always_inline)) uint32_t stamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}

extern "C" void dmab_run(uint32_t *ring, uint32_t *px, uint32_t *py, const int32_t *arg, uint32_t *res) {
  const int mode = arg[0];
  uint32_t *p = (mode == 0 || mode == 2) ? px : py;
  acquire_greater_equal(ring, 48 + 7, 1);  // the first slot arrived: the DMA runs
  const uint32_t t0 = stamp();
  uint32_t tp = t0;
  for (int s = 0; s < NSEG; s++) {  // during the DMA (NSEG x 256 cycles < its ~8,000)
    if (mode <= 1) dmab_ld(p, TRIPS);
    else if (mode <= 3) dmab_st(p, TRIPS);
    else dmab_nop(p, TRIPS);
    const uint32_t t = stamp();
    res[8 + s] = t - tp;
    tp = t;
  }
  const uint32_t tl = stamp();
  acquire_greater_equal(ring, 48 + 7, 62);  // all 63 slots written
  const uint32_t te = stamp();
  tp = te;
  for (int s = 0; s < NSEG; s++) {  // the same without the DMA
    if (mode <= 1) dmab_ld(p, TRIPS);
    else if (mode <= 3) dmab_st(p, TRIPS);
    else dmab_nop(p, TRIPS);
    const uint32_t t = stamp();
    res[8 + NSEG + s] = t - tp;
    tp = t;
  }
  res[3] = te - tl;  // the wait for the DMA's end after the segments (> 0: all segments overlapped it)
  release(ring, 48 + 6, 63);  // the free lock back to 63 for the next dispatch
  res[0] = 0;
  res[1] = te - t0;
  res[2] = mode;
}
