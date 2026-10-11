// round 19: one-core bench of the B pop-stream wrap probes (r19/wrapgen.py): source byte i = i mod 251 (4,608 B, bank 2), the
// probe stores 2 x 64 panels (72 B each); y[4 k ..] = (sum, sum of squares) of the x bytes and of the e bytes of pop k.
#include <aie_api/aie.hpp>
#include <stdint.h>
#define P(n) extern "C" void probe##n(const void *in, void *out, int n_, const float *, const float *);
P(0) P(1) P(2) P(3)
#define SEC(b) __attribute__((section(".aie.bank" #b)))
SEC(2) alignas(64) static uint8_t g_a[4608 + 1024];
alignas(64) static uint8_t g_o[8384 + 1024 + 64];
extern "C" void apb(int32_t *x, int32_t *y) {
  for (int i = 0; i < 4608 + 1024; i++) g_a[i] = (uint8_t)(i % 251);
  for (int i = 0; i < 8384 + 1024 + 64; i++) g_o[i] = 0xEE;
  switch (x[0]) {
    case 1: probe1(g_a, g_o, 0, 0, 0); break;
    case 2: probe2(g_a, g_o, 0, 0, 0); break;
    case 3: probe3(g_a, g_o, 0, 0, 0); break;
    default: probe0(g_a, g_o, 0, 0, 0);
  }
  for (int k = 0; k < 128; k++) {
    int32_t sx = 0, qx = 0, se = 0, qe = 0;
    for (int i = 0; i < 64; i++) { int v = g_o[64 * k + i]; sx += v; qx += v * v; }
    for (int i = 0; i < 8; i++) { int v = g_o[8384 + 8 * k + i]; se += v; qe += v * v; }
    y[4 * k] = sx; y[4 * k + 1] = qx; y[4 * k + 2] = se; y[4 * k + 3] = qe;
  }
  y[512] = g_o[8192] | (g_o[8384 + 1024] << 8);  // guard byte after the last panel (0xEE)
}
