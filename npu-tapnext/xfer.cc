// Transfer probe kernels: xfer_fin zeroes the result object; xfer_sum folds one input object into it
// (vector xor over 16-word slices) so the host can check every byte went through the core.
#include <stdint.h>

#include <aie_api/aie.hpp>

extern "C" {
#ifdef FIN
void xfer_fin(uint32_t *res) {
  for (int i = 0; i < OUT / 16; i++) aie::store_v(res + 16 * i, aie::zeros<uint32_t, 16>());
}
#else
void xfer_sum(uint32_t *in, uint32_t *res) {
  for (int i = 0; i < OBJ / 16; i++) {
    const int j = i % (OUT / 16);
    aie::vector<uint32_t, 16> a = aie::load_v<16>(res + 16 * j);
    aie::vector<uint32_t, 16> b = aie::load_v<16>(in + 16 * i);
    aie::store_v(res + 16 * j, aie::bit_xor(a, b));
  }
}
#endif
#ifdef AT
void xfer_sum_at(uint32_t *ring, uint32_t *res, int32_t slot) {
  uint32_t *in = ring + slot * OBJ;
  for (int i = 0; i < OBJ / 16; i++) {
    const int j = i % (OUT / 16);
    aie::vector<uint32_t, 16> a = aie::load_v<16>(res + 16 * j);
    aie::vector<uint32_t, 16> b = aie::load_v<16>(in + 16 * i);
    aie::store_v(res + 16 * j, aie::bit_xor(a, b));
  }
}
#endif
}
