// hipBLASLt int8 GEMM probe for gfx1151: enumerate all algos for D = X(MxK,int8) * W(NxK,int8)^T, time each, report best.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#define CK(x) do { auto e = (x); if (e != hipSuccess) { printf("HIP error %d at %d\n", (int)e, __LINE__); exit(1);} } while (0)
#define CB(x) do { auto e = (x); if (e != HIPBLAS_STATUS_SUCCESS) { printf("hipblaslt error %d at %d\n", (int)e, __LINE__); exit(1);} } while (0)

static const char* tname(hipDataType t) {
    switch (t) { case HIP_R_32I: return "i32"; case HIP_R_32F: return "f32"; case HIP_R_16BF: return "bf16"; case HIP_R_16F: return "f16"; case HIP_R_8I: return "i8"; default: return "?"; }
}

int main(int argc, char** argv) {
    int reps = 20;
    hipblasLtHandle_t h; CB(hipblasLtCreate(&h));
    hipStream_t st; CK(hipStreamCreate(&st));
    struct S { int M, K, N; };
    std::vector<S> shapes = {{730, 2048, 6144}, {730, 2048, 2048}, {730, 2048, 8192}, {730, 8192, 2048}, {32, 2048, 8192}, {32, 8192, 2048}};
    size_t wsz = 64u << 20; void* ws; CK(hipMalloc(&ws, wsz));
    for (auto s : shapes) {
        int M = s.M, K = s.K, N = s.N;
        int8_t *X, *W; CK(hipMalloc(&X, (size_t)M * K)); CK(hipMalloc(&W, (size_t)N * K));
        std::vector<int8_t> hx((size_t)M * K), hw((size_t)N * K);
        for (auto& v : hx) v = (int8_t)(rand() % 255 - 127);
        for (auto& v : hw) v = (int8_t)(rand() % 255 - 127);
        CK(hipMemcpy(X, hx.data(), hx.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(W, hw.data(), hw.size(), hipMemcpyHostToDevice));
        void* D; CK(hipMalloc(&D, (size_t)M * N * 4));
        float *sa, *sb; CK(hipMalloc(&sa, M * 4)); CK(hipMalloc(&sb, N * 4));
        std::vector<float> hs(std::max(M, N), 0.01f);
        CK(hipMemcpy(sa, hs.data(), M * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(sb, hs.data(), N * 4, hipMemcpyHostToDevice));
        for (hipDataType dt : {HIP_R_32I, HIP_R_8I, HIP_R_32F, HIP_R_16BF, HIP_R_16F}) {
            for (int scalemode = 0; scalemode < 2; scalemode++) {
                if (scalemode && dt == HIP_R_32I) continue;
                hipblasLtMatmulDesc_t md; CB(hipblasLtMatmulDescCreate(&md, HIPBLAS_COMPUTE_32I, HIP_R_32I));
                hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
                CB(hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
                CB(hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
                if (scalemode) {
                    hipblasLtMatmulMatrixScale_t sm = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
                    auto e1 = hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
                    auto e2 = hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));
                    auto e3 = hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sb, sizeof(sb));  // A' = W -> per-N scale
                    auto e4 = hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sa, sizeof(sa));
                    if (e1 || e2 || e3 || e4) { printf("M=%d K=%d N=%d D=%s vecscale: desc set failed %d %d %d %d\n", M, K, N, tname(dt), (int)e1, (int)e2, (int)e3, (int)e4); hipblasLtMatmulDescDestroy(md); continue; }
                }
                hipblasLtMatrixLayout_t la, lb, lc;
                CB(hipblasLtMatrixLayoutCreate(&la, HIP_R_8I, K, N, K));   // W col-major KxN, op T -> NxK
                CB(hipblasLtMatrixLayoutCreate(&lb, HIP_R_8I, K, M, K));   // X col-major KxM
                CB(hipblasLtMatrixLayoutCreate(&lc, dt, N, M, N));         // D col-major NxM == row-major MxN
                std::vector<hipblasLtMatmulHeuristicResult_t> algos;
                auto st_all = hipblaslt_ext::getAllAlgos(h, hipblaslt_ext::GemmType::HIPBLASLT_GEMM, ta, tb, HIP_R_8I, HIP_R_8I, dt, dt, HIPBLAS_COMPUTE_32I, algos);
                int32_t alpha = 1, beta = 0; float alphaf = 1.f, betaf = 0.f;
                const void* pa = (dt == HIP_R_32I || dt == HIP_R_8I) ? (const void*)&alpha : (const void*)&alphaf;
                const void* pb = (dt == HIP_R_32I || dt == HIP_R_8I) ? (const void*)&beta : (const void*)&betaf;
                int nsup = 0; float best = 1e9; int besti = -1; std::string bestname;
                for (size_t i = 0; i < algos.size(); i++) {
                    size_t need = 0;
                    if (hipblaslt_ext::matmulIsAlgoSupported(h, md, pa, la, lb, pb, lc, lc, algos[i].algo, need) != HIPBLAS_STATUS_SUCCESS || need > wsz) continue;
                    nsup++;
                    auto run = [&]() { return hipblasLtMatmul(h, md, pa, W, la, X, lb, pb, D, lc, D, lc, &algos[i].algo, ws, wsz, st); };
                    if (run() != HIPBLAS_STATUS_SUCCESS) { nsup--; continue; }
                    run(); CK(hipStreamSynchronize(st));
                    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
                    hipEventRecord(e0, st); for (int r = 0; r < reps; r++) run(); hipEventRecord(e1, st); hipEventSynchronize(e1);
                    float ms; hipEventElapsedTime(&ms, e0, e1); ms /= reps;
                    if (ms < best) { best = ms; besti = (int)i; bestname = hipblaslt_ext::getSolutionNameFromAlgo(h, algos[i].algo); }
                    hipEventDestroy(e0); hipEventDestroy(e1);
                }
                double tops = 2.0 * M * N * K / (best * 1e-3) / 1e12;
                printf("M=%d K=%d N=%d D=%s vecscale=%d: getAllAlgos=%d (%zu algos) supported=%d best=%.1f us (%.1f TOPS) idx=%d %s\n", M, K, N, tname(dt), scalemode, (int)st_all, algos.size(), nsup, best * 1e3, nsup ? tops : 0.0, besti, bestname.c_str());
                fflush(stdout);
                hipblasLtMatrixLayoutDestroy(la); hipblasLtMatrixLayoutDestroy(lb); hipblasLtMatrixLayoutDestroy(lc); hipblasLtMatmulDescDestroy(md);
            }
        }
        // bf16 reference for the same shapes (what rocBLAS/hipBLASLt bf16 gives)
        {
            hipblasLtMatmulDesc_t md; CB(hipblasLtMatmulDescCreate(&md, HIPBLAS_COMPUTE_32F, HIP_R_32F));
            hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
            CB(hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
            CB(hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
            hipblasLtMatrixLayout_t la, lb, lc;
            CB(hipblasLtMatrixLayoutCreate(&la, HIP_R_16BF, K, N, K)); CB(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16BF, K, M, K)); CB(hipblasLtMatrixLayoutCreate(&lc, HIP_R_16BF, N, M, N));
            void *Xb, *Wb; CK(hipMalloc(&Xb, (size_t)M * K * 2)); CK(hipMalloc(&Wb, (size_t)N * K * 2));
            std::vector<hipblasLtMatmulHeuristicResult_t> algos;
            hipblaslt_ext::getAllAlgos(h, hipblaslt_ext::GemmType::HIPBLASLT_GEMM, ta, tb, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F, algos);
            float alpha = 1, beta = 0; int nsup = 0; float best = 1e9; std::string bestname;
            for (size_t i = 0; i < algos.size(); i++) {
                size_t need = 0;
                if (hipblaslt_ext::matmulIsAlgoSupported(h, md, &alpha, la, lb, &beta, lc, lc, algos[i].algo, need) != HIPBLAS_STATUS_SUCCESS || need > wsz) continue;
                auto run = [&]() { return hipblasLtMatmul(h, md, &alpha, Wb, la, Xb, lb, &beta, D, lc, D, lc, &algos[i].algo, ws, wsz, st); };
                if (run() != HIPBLAS_STATUS_SUCCESS) continue;
                nsup++; run(); CK(hipStreamSynchronize(st));
                hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
                hipEventRecord(e0, st); for (int r = 0; r < reps; r++) run(); hipEventRecord(e1, st); hipEventSynchronize(e1);
                float ms; hipEventElapsedTime(&ms, e0, e1); ms /= reps;
                if (ms < best) { best = ms; bestname = hipblaslt_ext::getSolutionNameFromAlgo(h, algos[i].algo); }
                hipEventDestroy(e0); hipEventDestroy(e1);
            }
            printf("M=%d K=%d N=%d bf16 ref: %zu algos supported=%d best=%.1f us (%.1f TFLOPS) %s\n", M, K, N, algos.size(), nsup, best * 1e3, 2.0 * M * N * K / (best * 1e-3) / 1e12, bestname.c_str());
            hipFree(Xb); hipFree(Wb);
        }
        hipFree(X); hipFree(W); hipFree(D); hipFree(sa); hipFree(sb);
    }
    return 0;
}
