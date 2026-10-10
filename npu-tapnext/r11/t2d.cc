// round 11: how Peano encodes 2D post-modify addressing (asm inspection only)
#include <aie_api/aie.hpp>
extern "C" void t2d(const bfloat16 *__restrict in, bfloat16 *__restrict out, int n) {
  int c1 = 0, c2 = 0;
  for (int i = 0; i < n; i++) {
    aie::vector<bfloat16, 32> v = aie::load_v<32>(in);
    in = add_2d_byte(in, 576, 7, c1, 64);  // 8 vectors of 64 B, then +576 (skip 512 B)
    aie::store_v(out, v);
    out = add_2d_byte(out, 1088, 3, c2, 128);
  }
}
