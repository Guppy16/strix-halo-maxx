// pass 19 MULLAT bench (r24/mullat_gen.py): probe k stores the r26 values its stores saw
#include <aie_api/aie.hpp>
#include <stdint.h>
#define P(n) extern "C" void probe##n(const void *in, void *out, int n_, const float *, const float *);
P(0) P(1) P(2) P(3) P(4) P(5) P(6) P(7)
alignas(64) static int32_t g_o[32];
extern "C" void apb(int32_t *x, int32_t *y) {
  for (int i = 0; i < 32; i++) g_o[i] = -1;
  switch (x[0]) {
    case 1: probe1(0, g_o, 0, 0, 0); break;
    case 2: probe2(0, g_o, 0, 0, 0); break;
    case 3: probe3(0, g_o, 0, 0, 0); break;
    case 4: probe4(0, g_o, 0, 0, 0); break;
    case 5: probe5(0, g_o, 0, 0, 0); break;
    case 6: probe6(0, g_o, 0, 0, 0); break;
    case 7: probe7(0, g_o, 0, 0, 0); break;
    default: probe0(0, g_o, 0, 0, 0);
  }
  for (int i = 0; i < 32; i++) y[i] = g_o[i];
}
