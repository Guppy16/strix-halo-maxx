
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
extern "C" int bf16gemm_launch(int cfg, hipStream_t st, const uint16_t* A, const uint16_t* W, const float* bias, uint16_t* C, int M, int N, int K, int gelu, int swap, int lda, int ldw, int packed);
extern "C" int bf16gemm_ncfg();
#define CK(x) do { auto e = (x); if (e != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1);} } while (0)
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return u >> 16; }
static float gelu(float x) { float u = 0.7978845608028654f * (x + 0.044715f * x * x * x); return 0.5f * x * (1.0f + tanhf(u)); }
int main(int argc, char** argv) {
    hipStream_t st; CK(hipStreamCreate(&st));
    int reps = 20;
    struct S { int M, K, N; };
    std::vector<S> shapes = {{729, 1152, 3456}, {729, 1152, 1152}, {729, 1152, 4304}, {729, 4304, 1152}, {729, 2304, 8192}, {729, 8192, 2048}, {730, 2048, 6144}};
    int only = argc > 1 ? atoi(argv[1]) : -1;
    for (auto s : shapes) {
        int M = s.M, K = s.K, N = s.N;
        std::vector<uint16_t> hA((size_t)M * K), hW((size_t)N * K); std::vector<float> fA(hA.size()), fW(hW.size()), hb(N);
        for (size_t i = 0; i < hA.size(); i++) { hA[i] = f2bf((rand() % 2000 - 1000) / 1000.f); fA[i] = bf2f(hA[i]); }
        for (size_t i = 0; i < hW.size(); i++) { hW[i] = f2bf((rand() % 2000 - 1000) / 20000.f); fW[i] = bf2f(hW[i]); }
        for (auto& v : hb) v = (rand() % 200 - 100) / 100.f;
        uint16_t *A, *W, *C; float* b;
        CK(hipMalloc(&A, hA.size() * 2)); CK(hipMalloc(&W, hW.size() * 2)); CK(hipMalloc(&b, N * 4)); CK(hipMalloc(&C, (size_t)M * N * 2));
        CK(hipMemcpy(A, hA.data(), hA.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(W, hW.data(), hW.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(b, hb.data(), N * 4, hipMemcpyHostToDevice));
        std::vector<int> rows; for (int i = 0; i < 16; i++) rows.push_back(rand() % M); rows.push_back(M - 1); rows.push_back(0);
        std::vector<float> ref(rows.size() * N);
        for (size_t ri = 0; ri < rows.size(); ri++) { int m = rows[ri]; for (int n = 0; n < N; n++) { double acc = 0; const float* a = &fA[(size_t)m * K]; const float* w = &fW[(size_t)n * K]; for (int k = 0; k < K; k++) acc += (double)a[k] * w[k]; ref[ri * N + n] = (float)acc + hb[n]; } }
        std::vector<uint16_t> hC((size_t)M * N);
        for (int cfg = 0; cfg < bf16gemm_ncfg(); cfg++) {
            if (only >= 0 && cfg != only) continue;
            for (int gs = 0; gs < 4; gs++) { int g = gs & 1, swap = gs >> 1; if (g && !getenv("GELU")) continue;
                CK(hipMemsetAsync(C, 0, (size_t)M * N * 2, st));
                int rc = bf16gemm_launch(cfg, st, A, W, b, C, M, N, K, g, swap, K, K, 0);
                if (rc) { printf("M=%d K=%d N=%d cfg=%d: unsupported (%d)\n", M, K, N, cfg, rc); continue; }
                CK(hipStreamSynchronize(st)); CK(hipGetLastError());
                CK(hipMemcpy(hC.data(), C, hC.size() * 2, hipMemcpyDeviceToHost));
                double maxabs = 0, maxrel = 0;
                for (size_t ri = 0; ri < rows.size(); ri++) for (int n = 0; n < N; n++) { float r = ref[ri * N + n]; if (g) r = gelu(r); float o = bf2f(hC[(size_t)rows[ri] * N + n]); double d = fabs(o - r); maxabs = std::max(maxabs, d); maxrel = std::max(maxrel, d / (fabs(r) + 1e-1)); }
                hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
                for (int r = 0; r < 3; r++) bf16gemm_launch(cfg, st, A, W, b, C, M, N, K, g, swap, K, K, 0);
                CK(hipEventRecord(e0, st)); for (int r = 0; r < reps; r++) bf16gemm_launch(cfg, st, A, W, b, C, M, N, K, g, swap, K, K, 0); CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1)); ms /= reps;
                printf("bf16 M=%d K=%d N=%d cfg=%d gelu=%d swap=%d: %.1f us (%.1f TFLOPS) maxabs=%.4f maxrel=%.4f %s\n", M, K, N, cfg, g, swap, ms * 1e3, 2.0 * M * N * K / (ms * 1e-3) / 1e12, maxabs, maxrel, maxrel < 0.02 ? "OK" : "MISMATCH");
                fflush(stdout);
            }
        }
        CK(hipFree(A)); CK(hipFree(W)); CK(hipFree(b)); CK(hipFree(C));
    }
    return 0;
}
