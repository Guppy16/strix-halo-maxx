// round 22 KTLP-lite: scalar lda.2d (KTL's record wrap) on one core; probe k stores the 15 words it loaded
#include <aie_api/aie.hpp>
#include <stdint.h>
#define P(n) extern "C" void probe##n(const void *in, void *out, int n_, const float *, const float *);
P(0) P(1) P(2) P(3) P(4)
alignas(64) static uint8_t g_a[256];
alignas(64) static int32_t g_o[32];
extern "C" void apb(int32_t *x, int32_t *y) {
  for (int i = 0; i < 256; i++) g_a[i] = (uint8_t)(i * 7 + 1);
  for (int i = 0; i < 32; i++) g_o[i] = -1;
  switch (x[0]) {
    case 1: probe1(g_a, g_o, 0, 0, 0); break;
    case 2: probe2(g_a, g_o, 0, 0, 0); break;
    case 3: probe3(g_a, g_o, 0, 0, 0); break;
    case 4: probe4(g_a, g_o, 0, 0, 0); break;
    default: probe0(g_a, g_o, 0, 0, 0);
  }
  for (int i = 0; i < 32; i++) y[i] = g_o[i];
}
