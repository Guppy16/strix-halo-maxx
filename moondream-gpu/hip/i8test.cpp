// correctness + timing harness for i8gemm.hip
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
extern "C" int i8gemm_launch(int cfg, hipStream_t st, const int8_t* A, const int8_t* W, const float* sa, const float* sw, const float* bias, uint16_t* C, int M, int N, int K, int gelu, int swap, int lda, int ldw, int packed);
extern "C" int i8gemm_ncfg();
extern "C" int i8gemm_small_launch(int cfg, hipStream_t st, const int8_t* A, const int8_t* W, int32_t* P, const float* sa, const float* sw, const float* bias, uint16_t* C, int M, int N, int K, int gelu, int lda, int ldw, int packed);
extern "C" int i8gemm_small_ncfg();
extern "C" void wmma_peak(hipStream_t st, int blocks, int threads, int iters, int* out);
extern "C" void wmma_peak_bf16(hipStream_t st, int blocks, int threads, int iters, int* out);
#define CK(x) do { auto e = (x); if (e != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1);} } while (0)
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static float gelu(float x) { float u = 0.7978845608028654f * (x + 0.044715f * x * x * x); return 0.5f * x * (1.0f + tanhf(u)); }
int main(int argc, char** argv) {
    hipStream_t st; CK(hipStreamCreate(&st));
    int reps = 20;
    { // peak probe
        int* o; CK(hipMalloc(&o, 4));
        for (int blocks : {40, 80, 160, 320}) {
            wmma_peak(st, blocks, 256, 1000, o); CK(hipStreamSynchronize(st));
            hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
            CK(hipEventRecord(e0, st)); wmma_peak(st, blocks, 256, 20000, o); CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1));
            double ops = 2.0 * 16 * 16 * 16 * 8.0 * 20000 * blocks * 8;  // 8 waves per block
            printf("wmma int8 peak probe: %d blocks x 8 waves: %.1f TOPS\n", blocks, ops / (ms * 1e-3) / 1e12);
            wmma_peak_bf16(st, blocks, 256, 1000, o); CK(hipStreamSynchronize(st));
            CK(hipEventRecord(e0, st)); wmma_peak_bf16(st, blocks, 256, 20000, o); CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
            CK(hipEventElapsedTime(&ms, e0, e1));
            printf("wmma bf16 peak probe: %d blocks x 8 waves: %.1f TFLOPS\n", blocks, ops / (ms * 1e-3) / 1e12);
        }
    }
    struct S { int M, K, N; };
    std::vector<S> shapes = {{730, 2048, 6144}, {730, 2048, 2048}, {730, 2048, 8192}, {730, 8192, 2048}, {32, 2048, 8192}, {32, 8192, 2048}};
    if (getenv("SMALL")) shapes = {{32, 2048, 6144}, {32, 2048, 2048}, {32, 2048, 8192}, {32, 8192, 2048}, {64, 2048, 8192}, {64, 8192, 2048}};
    if (getenv("L2TEST")) shapes = {{4096, 2048, 1024}, {4096, 1024, 2048}, {2048, 2048, 2048}, {730, 2048, 6144}};
    int only = argc > 1 ? atoi(argv[1]) : -1; int cfg_from = getenv("CFG_FROM") ? atoi(getenv("CFG_FROM")) : 0;
    for (auto s : shapes) {
        int M = s.M, K = s.K, N = s.N;
        std::vector<int8_t> hA((size_t)M * K), hW((size_t)N * K); std::vector<float> hsa(M), hsw(N), hb(N);
        for (auto& v : hA) v = (int8_t)(rand() % 255 - 127);
        for (auto& v : hW) v = (int8_t)(rand() % 255 - 127);
        for (auto& v : hsa) v = 0.005f + 0.01f * (rand() % 100) / 100.f;
        for (auto& v : hsw) v = 0.0005f + 0.001f * (rand() % 100) / 100.f;
        for (auto& v : hb) v = (rand() % 200 - 100) / 100.f;
        int8_t *A, *W; float *sa, *sw, *b; uint16_t* C;
        CK(hipMalloc(&A, hA.size())); CK(hipMalloc(&W, hW.size())); CK(hipMalloc(&sa, M * 4)); CK(hipMalloc(&sw, N * 4)); CK(hipMalloc(&b, N * 4)); CK(hipMalloc(&C, (size_t)M * N * 2));
        CK(hipMemcpy(A, hA.data(), hA.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(W, hW.data(), hW.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(sa, hsa.data(), M * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(sw, hsw.data(), N * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(b, hb.data(), N * 4, hipMemcpyHostToDevice));
        // reference on a sample of rows
        std::vector<int> rows; for (int i = 0; i < 24; i++) rows.push_back(rand() % M); rows.push_back(M - 1); rows.push_back(0);
        std::vector<float> ref(rows.size() * N);
        for (size_t ri = 0; ri < rows.size(); ri++) { int m = rows[ri]; for (int n = 0; n < N; n++) { long acc = 0; const int8_t* a = &hA[(size_t)m * K]; const int8_t* w = &hW[(size_t)n * K]; for (int k = 0; k < K; k++) acc += (int)a[k] * (int)w[k]; ref[ri * N + n] = (float)acc * hsa[m] * hsw[n] + hb[n]; } }
        std::vector<uint16_t> hC((size_t)M * N);
        if (M <= 64) {
            int32_t* P; CK(hipMalloc(&P, (size_t)16 * 64 * 8192 * 4));
            for (int cfg = 0; cfg < i8gemm_small_ncfg(); cfg++) {
                CK(hipMemsetAsync(C, 0, (size_t)M * N * 2, st));
                int rc = i8gemm_small_launch(cfg, st, A, W, P, sa, sw, b, C, M, N, K, 0, K, K, 0);
                if (rc) { printf("small M=%d K=%d N=%d cfg=%d: unsupported (%d)\n", M, K, N, cfg, rc); continue; }
                CK(hipStreamSynchronize(st)); CK(hipGetLastError());
                CK(hipMemcpy(hC.data(), C, hC.size() * 2, hipMemcpyDeviceToHost));
                double maxabs = 0, maxrel = 0;
                for (size_t ri = 0; ri < rows.size(); ri++) for (int n = 0; n < N; n++) { float r = ref[ri * N + n]; float o = bf2f(hC[(size_t)rows[ri] * N + n]); double d = fabs(o - r); maxabs = std::max(maxabs, d); maxrel = std::max(maxrel, d / (fabs(r) + 1e-2)); }
                hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
                for (int r = 0; r < 3; r++) i8gemm_small_launch(cfg, st, A, W, P, sa, sw, b, C, M, N, K, 0, K, K, 0);
                CK(hipEventRecord(e0, st)); for (int r = 0; r < reps; r++) i8gemm_small_launch(cfg, st, A, W, P, sa, sw, b, C, M, N, K, 0, K, K, 0); CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1)); ms /= reps;
                printf("small M=%d K=%d N=%d cfg=%d: %.1f us (%.1f TOPS, %.0f GB/s weights) maxabs=%.4f maxrel=%.4f %s\n", M, K, N, cfg, ms * 1e3, 2.0 * M * N * K / (ms * 1e-3) / 1e12, (double)N * K / (ms * 1e-3) / 1e9, maxabs, maxrel, maxrel < 0.02 ? "OK" : "MISMATCH");
                fflush(stdout);
            }
            CK(hipFree(P));
            if (getenv("SMALL")) continue;
        }
        for (int cfg = 0; cfg < i8gemm_ncfg(); cfg++) {
            if (only >= 0 && cfg != only) continue; if (cfg < cfg_from) continue;
            for (int gs = 0; gs < 4; gs++) { int g = gs & 1, swap = gs >> 1; if (g && !getenv("GELU")) continue; if (getenv("SWAP1") && !swap) continue;
                CK(hipMemsetAsync(C, 0, (size_t)M * N * 2, st));
                int rc = i8gemm_launch(cfg, st, A, W, sa, sw, b, C, M, N, K, g, swap, K, K, 0);
                if (rc) { printf("M=%d K=%d N=%d cfg=%d gelu=%d: unsupported (%d)\n", M, K, N, cfg, g, rc); continue; }
                CK(hipStreamSynchronize(st)); CK(hipGetLastError());
                CK(hipMemcpy(hC.data(), C, hC.size() * 2, hipMemcpyDeviceToHost));
                double maxabs = 0, maxrel = 0;
                for (size_t ri = 0; ri < rows.size(); ri++) for (int n = 0; n < N; n++) { float r = ref[ri * N + n]; if (g) r = gelu(r); float o = bf2f(hC[(size_t)rows[ri] * N + n]); double d = fabs(o - r); maxabs = std::max(maxabs, d); maxrel = std::max(maxrel, d / (fabs(r) + 1e-2)); }
                hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
                for (int r = 0; r < 3; r++) i8gemm_launch(cfg, st, A, W, sa, sw, b, C, M, N, K, g, swap, K, K, 0);
                CK(hipEventRecord(e0, st)); for (int r = 0; r < reps; r++) i8gemm_launch(cfg, st, A, W, sa, sw, b, C, M, N, K, g, swap, K, K, 0); CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1)); ms /= reps;
                printf("M=%d K=%d N=%d cfg=%d gelu=%d swap=%d: %.1f us (%.1f TOPS) maxabs=%.4f maxrel=%.4f %s\n", M, K, N, cfg, g, swap, ms * 1e3, 2.0 * M * N * K / (ms * 1e-3) / 1e12, maxabs, maxrel, maxrel < 0.02 ? "OK" : "MISMATCH");
                fflush(stdout);
            }
        }
        CK(hipFree(A)); CK(hipFree(W)); CK(hipFree(sa)); CK(hipFree(sw)); CK(hipFree(b)); CK(hipFree(C));
    }
    return 0;
}
