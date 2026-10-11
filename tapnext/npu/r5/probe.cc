#include <stdint.h>

#include <aie_api/aie.hpp>

using B32 = aie::vector<bfloat16, 32>;
using F32 = aie::vector<float, 32>;
using A32 = aie::accum<accfloat, 32>;
#define BF(e) ((e).template to_vector<bfloat16>())
#define FV(e) ((e).template to_vector<float>())

// every lane of [4 rows][8] gets its row's sum
static inline A32 rowsum8(A32 a) {
  for (int st = 4; st >= 1; st /= 2) {
    F32 v = FV(a);
    auto [e, o] = aie::interleave_unzip(v, v, st);
    F32 sw = aie::interleave_zip(o, e, st).first;
    a = aie::add(a, A32(sw));
  }
  return a;
}

extern "C" void probe_rowsum(const float *in, float *out) {
  A32 a(aie::load_v<32>(in));
  aie::store_v(out, FV(rowsum8(a)));
}

static inline A32 rowsum8b(A32 a) {  // on bf16 hi/lo pairs
  const B32 ones = aie::broadcast<bfloat16, 32>(1.0f);
  for (int st = 4; st >= 1; st /= 2) {
    B32 h = BF(a);
    B32 l = BF(aie::mac(a, h, (bfloat16)-1.0f));
    auto [eh, oh] = aie::interleave_unzip(h, h, st);
    auto [el, ol] = aie::interleave_unzip(l, l, st);
    a = aie::mac(a, aie::interleave_zip(oh, eh, st).first, ones);
    a = aie::mac(a, aie::interleave_zip(ol, el, st).first, ones);
  }
  return a;
}

extern "C" void probe_rowsum_b(const float *in, float *out) {
  A32 a(aie::load_v<32>(in));
  aie::store_v(out, FV(rowsum8b(a)));
}
