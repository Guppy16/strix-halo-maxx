// round 10: runtime-task A slot bench (r10/rtm.py). The core drains its A ring (3 stripes of 288 words, locks aF = 4 /
// aU = 5 as design8) and hashes every stripe: h = 31 h + sum(words). Per op: hash, cycles from the previous op's end to
// its first stripe (the op-start wait), cycles of its stripes.
#include <aie_api/aie.hpp>

static inline __attribute__((always_inline)) uint32_t stamp() {
  __builtin_aie2p_sched_barrier();
  const uint32_t v = read_tm(0x340F8);
  __builtin_aie2p_sched_barrier();
  return v;
}

extern "C" void rtm_run(uint32_t *a0, uint32_t *a1, uint32_t *a2, const int32_t *tab, uint32_t *res) {
  uint32_t *slot[3] = {a0, a1, a2};
  const int nops = tab[0];
  int j = 0;
  uint32_t tprev = stamp();
  for (int o = 0; o < nops; o++) {
    const int ns = tab[1 + o];
    uint32_t h = 0, t0 = 0;
    for (int s = 0; s < ns; s++) {
      acquire_greater_equal(slot[j], 48 + 5, 1);
      if (s == 0) t0 = stamp();
      aie::vector<int32, 16> v = aie::zeros<int32, 16>();
      const int32_t *p = (const int32_t *)slot[j];
      for (int i = 0; i < 18; i++) v = aie::add(v, aie::load_v<16>(p + 16 * i));
      release(slot[j], 48 + 4, 1);
      h = h * 31u + (uint32_t)aie::reduce_add(v);
      j = j == 2 ? 0 : j + 1;
    }
    const uint32_t t1 = stamp();
    res[3 * o] = h;
    res[3 * o + 1] = t0 - tprev;
    res[3 * o + 2] = t1 - t0;
    tprev = t1;
  }
}
