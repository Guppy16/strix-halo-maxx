// round 16: one-core probe of a restarted 576 B vlda.pop.576 stream (r16/popprobe.s): the buffer holds byte i = i (mod 251)
#include <aie_api/aie.hpp>
#include <stdint.h>
extern "C" void probe0(const void *in, void *out, int n, const float *, const float *);
extern "C" void probe1(const void *in, void *out, int n, const float *, const float *);
extern "C" void probe6(const void *in, void *out, int n, const float *, const float *);
extern "C" void probe7(const void *in, void *out, int n, const float *, const float *);
extern "C" void probe5(const void *in, void *out, int n, const float *, const float *);
extern "C" void probe4(const void *in, void *out, int n, const float *, const float *);
extern "C" void probe3(const void *in, void *out, int n, const float *, const float *);
extern "C" void probe2(const void *in, void *out, int n, const float *, const float *);
#define SEC(b) __attribute__((section(".aie.bank" #b)))
SEC(2) alignas(64) static uint8_t g_a[2048 + 1024];
alignas(64) static uint8_t g_o[1024];
extern "C" void apb(int32_t *x, int32_t *y) {
  for (int i = 0; i < 3072; i++) g_a[i] = (uint8_t)(i % 251);
  for (int i = 0; i < 1024; i++) g_o[i] = 0xEE;
  if (x[0] == 6) probe6(g_a, g_o, 0, 0, 0); else if (x[0] == 7) probe7(g_a, g_o, 0, 0, 0); else if (x[0] == 5) probe5(g_a, g_o, 0, 0, 0); else if (x[0] == 4) probe4(g_a, g_o, 0, 0, 0); else if (x[0] == 3) probe3(g_a, g_o, 0, 0, 0); else if (x[0] == 2) probe2(g_a, g_o, 0, 0, 0); else if (x[0] == 1) probe1(g_a, g_o, 0, 0, 0); else probe0(g_a, g_o, 0, 0, 0);
  for (int i = 0; i < 576 / 4; i++) y[i] = ((int32_t *)g_o)[i];
}
