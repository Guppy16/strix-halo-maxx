// round 16: Peano instruction forms for int16 MACs into acc32 (index / exponent arithmetic of the sqrt kernel)
#include <aie_api/aie.hpp>
using I16 = aie::vector<int16_t, 32>;
using A32i = aie::accum<acc32, 32>;
extern "C" {
__attribute__((noinline)) void p_mul(const int16_t *__restrict a, int32_t *__restrict o) {
  I16 x = aie::load_v<32>(a);
  A32i m = aie::mul(x, aie::broadcast<int16_t, 32>((int16_t)4));
  aie::store_v(o, m.template to_vector<int32_t>(0).template extract<16>(0));
  aie::store_v(o + 16, m.template to_vector<int32_t>(0).template extract<16>(1));
}
__attribute__((noinline)) void p_mac(const int16_t *__restrict a, const int32_t *__restrict base, int32_t *__restrict o) {
  I16 x = aie::load_v<32>(a);
  A32i b;
  b.from_vector(aie::load_v<32>(base), 0);
  A32i m = aie::mac(b, x, aie::broadcast<int16_t, 32>((int16_t)4));
  aie::store_v(o, m.template to_vector<int32_t>(0).template extract<16>(0));
  aie::store_v(o + 16, m.template to_vector<int32_t>(0).template extract<16>(1));
}
__attribute__((noinline)) void p_half(const int16_t *__restrict a, int16_t *__restrict o) {
  I16 x = aie::load_v<32>(a);
  A32i m = aie::mul(aie::bit_and(x, aie::broadcast<int16_t, 32>((int16_t)0xFF00)), aie::broadcast<int16_t, 32>((int16_t)1));
  aie::store_v(o, m.template to_vector<int16_t>(1));
}
}
