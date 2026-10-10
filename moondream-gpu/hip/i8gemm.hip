// int8 W8A8 GEMM for gfx11 (RDNA3/3.5) wave32 WMMA:  C[M,N] (bf16) = (A[M,K] int8 * W[N,K]^T int8) * sa[M] * sw[N] + bias[N] (+ gelu)
// A, W are K-contiguous (row-major), which is exactly the WMMA operand layout (lane l holds row l%16, 16 consecutive k).
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <type_traits>
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

__device__ __forceinline__ uint16_t f2bf(float f) { uint32_t u = __float_as_uint(f); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
__device__ __forceinline__ float gelu_tanh(float x) { float u = 0.7978845608028654f * (x + 0.044715f * x * x * x); float t = 1.0f - 2.0f / (1.0f + __expf(2.0f * u)); return 0.5f * x * (1.0f + t); }

template <int BM, int BN, int BK, int WM, int WN>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32)
i8gemm_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ W, const float* __restrict__ sa, const float* __restrict__ sw,
              const float* __restrict__ bias, uint16_t* __restrict__ C, int M, int N, int K, int gelu, int swap, int lda, int ldw) {
    constexpr int NW = (BM / WM) * (BN / WN), NT = NW * 32;
    constexpr int LDV = BK / 16 + 1;            // padded row stride in 16-byte units (BK+16 bytes): conflict-free ds_read_b128
    constexpr int CPR = BK / 16;                // 16-byte chunks per row
    constexpr int APT = BM * CPR / NT, BPT = BN * CPR / NT;
    static_assert(BM * CPR % NT == 0 && BN * CPR % NT == 0, "tile/threads mismatch");
    constexpr int FM = WM / 16, FN = WN / 16;
    __shared__ v4i As[2][BM * LDV];
    __shared__ v4i Bs[2][BN * LDV];
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int wm0 = (wid / (BN / WN)) * WM, wn0 = (wid % (BN / WN)) * WN;
    const int bm0 = (swap ? blockIdx.x : blockIdx.y) * BM, bn0 = (swap ? blockIdx.y : blockIdx.x) * BN;
    v4i ra[APT], rb[BPT];
    auto gload = [&](int kt) {
#pragma unroll
        for (int i = 0; i < APT; i++) { int c = tid + i * NT; int r = c / CPR, cc = c % CPR; int m = min(bm0 + r, M - 1); ra[i] = *(const v4i*)(A + (size_t)m * lda + kt * BK + cc * 16); }
#pragma unroll
        for (int i = 0; i < BPT; i++) { int c = tid + i * NT; int r = c / CPR, cc = c % CPR; int n = bn0 + r; rb[i] = *(const v4i*)(W + (size_t)n * ldw + kt * BK + cc * 16); }
    };
    auto lstore = [&](int buf) {
#pragma unroll
        for (int i = 0; i < APT; i++) { int c = tid + i * NT; int r = c / CPR, cc = c % CPR; As[buf][r * LDV + cc] = ra[i]; }
#pragma unroll
        for (int i = 0; i < BPT; i++) { int c = tid + i * NT; int r = c / CPR, cc = c % CPR; Bs[buf][r * LDV + cc] = rb[i]; }
    };
    v8i acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) acc[i][j] = (v8i){0, 0, 0, 0, 0, 0, 0, 0};
    const int KT = K / BK;
    gload(0); lstore(0); __syncthreads();
    for (int kt = 0; kt < KT; kt++) {
        const int buf = kt & 1;
        if (kt + 1 < KT) gload(kt + 1);
#pragma unroll
        for (int kk = 0; kk < CPR; kk++) {
            v4i af[FM], bf[FN];
#pragma unroll
            for (int i = 0; i < FM; i++) af[i] = As[buf][(wm0 + i * 16 + (lane & 15)) * LDV + kk];
#pragma unroll
            for (int j = 0; j < FN; j++) bf[j] = Bs[buf][(wn0 + j * 16 + (lane & 15)) * LDV + kk];
#pragma unroll
            for (int i = 0; i < FM; i++)
#pragma unroll
                for (int j = 0; j < FN; j++) acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[i], true, bf[j], acc[i][j], false);
        }
        if (kt + 1 < KT) lstore(buf ^ 1);
        __syncthreads();
    }
    // epilogue: acc element e of lane l is C[row = e*2 + l/16][col = l%16] of each 16x16 tile
    float swv[FN], bv[FN];
#pragma unroll
    for (int j = 0; j < FN; j++) { int n = bn0 + wn0 + j * 16 + (lane & 15); swv[j] = sw[n]; bv[j] = bias ? bias[n] : 0.f; }
#pragma unroll
    for (int i = 0; i < FM; i++) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
            int m = bm0 + wm0 + i * 16 + e * 2 + (lane >> 4);
            if (m < M) {
                float s = sa[m];
#pragma unroll
                for (int j = 0; j < FN; j++) {
                    int n = bn0 + wn0 + j * 16 + (lane & 15);
                    float v = (float)acc[i][j][e] * s * swv[j] + bv[j];
                    if (gelu) v = gelu_tanh(v);
                    C[(size_t)m * N + n] = f2bf(v);
                }
            }
        }
    }
}


// ---- v2: configurable LDS buffering (NBUF) and register prefetch distance (PD)
template <int BM, int BN, int BK, int WM, int WN, int NBUF, int PD, int PRIO, int MINB = 1, bool SPLIT = false>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32, MINB)
i8gemm2_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ W, const float* __restrict__ sa, const float* __restrict__ sw,
               const float* __restrict__ bias, uint16_t* __restrict__ C, int M, int N, int K, int gelu, int swap, int32_t* __restrict__ P, int kper, int lda, int ldw, int packed) {
    constexpr int NW = (BM / WM) * (BN / WN), NT = NW * 32;
    constexpr int LDV = BK / 16 + 1;
    constexpr int CPR = BK / 16;
    constexpr int APT = BM * CPR / NT, BPT = BN * CPR / NT;
    static_assert(BM * CPR % NT == 0 && BN * CPR % NT == 0, "tile/threads mismatch");
    constexpr int FM = WM / 16, FN = WN / 16;
    __shared__ v4i As[NBUF][BM * LDV];
    __shared__ v4i Bs[NBUF][BN * LDV];
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int wm0 = (wid / (BN / WN)) * WM, wn0 = (wid % (BN / WN)) * WN;
    const int bm0 = (swap ? blockIdx.x : blockIdx.y) * BM, bn0 = (swap ? blockIdx.y : blockIdx.x) * BN;
    v4i ra[PD][APT], rb[PD][BPT];
    const int k0 = SPLIT ? blockIdx.z * kper : 0;
    const int KT = (SPLIT ? kper : K) / BK;
    auto gload = [&](int kt, auto SET) {
        constexpr int s = decltype(SET)::value;
#pragma unroll
        for (int i = 0; i < APT; i++) { int c = tid + i * NT; int r = c / CPR, cc = c % CPR; int m = min(bm0 + r, M - 1); ra[s][i] = *(const v4i*)(A + (size_t)m * lda + k0 + kt * BK + cc * 16); }
#pragma unroll
        for (int i = 0; i < BPT; i++) { int c = tid + i * NT; int r = c / CPR, cc = c % CPR; int n = bn0 + r; rb[s][i] = packed ? *(const v4i*)(W + ((size_t)(bn0 / 128) * (K / BK) + (k0 / BK + kt)) * (128 * BK) + ((bn0 % 128) + r) * BK + cc * 16) : *(const v4i*)(W + (size_t)n * ldw + k0 + kt * BK + cc * 16); }
    };
    auto lstore = [&](auto SET, auto BUF) {
        constexpr int s = decltype(SET)::value, b = decltype(BUF)::value;
#pragma unroll
        for (int i = 0; i < APT; i++) { int c = tid + i * NT; int r = c / CPR, cc = c % CPR; As[b][r * LDV + cc] = ra[s][i]; }
#pragma unroll
        for (int i = 0; i < BPT; i++) { int c = tid + i * NT; int r = c / CPR, cc = c % CPR; Bs[b][r * LDV + cc] = rb[s][i]; }
    };
    v8i acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) acc[i][j] = (v8i){0, 0, 0, 0, 0, 0, 0, 0};
    auto compute = [&](auto BUF) {
        constexpr int b = decltype(BUF)::value;
        if (PRIO) __builtin_amdgcn_s_setprio(3);
#pragma unroll
        for (int kk = 0; kk < CPR; kk++) {
            v4i af[FM], bf[FN];
#pragma unroll
            for (int i = 0; i < FM; i++) af[i] = As[b][(wm0 + i * 16 + (lane & 15)) * LDV + kk];
#pragma unroll
            for (int j = 0; j < FN; j++) bf[j] = Bs[b][(wn0 + j * 16 + (lane & 15)) * LDV + kk];
#pragma unroll
            for (int i = 0; i < FM; i++)
#pragma unroll
                for (int j = 0; j < FN; j++) acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[i], true, bf[j], acc[i][j], false);
        }
        if (PRIO) __builtin_amdgcn_s_setprio(0);
    };
    using I0 = std::integral_constant<int, 0>; using I1 = std::integral_constant<int, 1>;
    // prologue
    gload(0, I0{});
    if (PD > 1 && KT > 1) gload(1, I1{});
    lstore(I0{}, I0{}); __syncthreads();
    auto step = [&](int kt, auto SET, auto BUF) {   // SET = set holding tile kt+1 (PD=2) / the only set (PD=1); BUF = LDS buffer of tile kt
        constexpr int s = decltype(SET)::value, b = decltype(BUF)::value;
        constexpr int sload = PD == 1 ? 0 : (s ^ 1);         // set that becomes free (held tile kt)
        constexpr int bnext = NBUF == 1 ? 0 : (b ^ 1);
        if (kt + PD < KT) gload(kt + PD, std::integral_constant<int, sload>{});
        compute(std::integral_constant<int, b>{});
        if (kt + 1 < KT) {
            if (NBUF == 1) __syncthreads();
            lstore(std::integral_constant<int, s>{}, std::integral_constant<int, bnext>{});
        }
        __syncthreads();
    };
    int kt = 0;
    for (; kt + 1 < KT; kt += 2) {
        step(kt, std::integral_constant<int, PD == 1 ? 0 : 1>{}, I0{});
        step(kt + 1, I0{}, std::integral_constant<int, NBUF == 1 ? 0 : 1>{});
    }
    if (kt < KT) step(kt, std::integral_constant<int, PD == 1 ? 0 : 1>{}, I0{});
    if constexpr (SPLIT) {
        int32_t* Pk = P + (size_t)blockIdx.z * M * N;
#pragma unroll
        for (int i = 0; i < FM; i++)
#pragma unroll
            for (int e = 0; e < 8; e++) {
                int m = bm0 + wm0 + i * 16 + e * 2 + (lane >> 4);
                if (m < M)
#pragma unroll
                    for (int j = 0; j < FN; j++) { int n = bn0 + wn0 + j * 16 + (lane & 15); Pk[(size_t)m * N + n] = acc[i][j][e]; }
            }
        return;
    }
    float swv[FN], bv[FN];
#pragma unroll
    for (int j = 0; j < FN; j++) { int n = bn0 + wn0 + j * 16 + (lane & 15); swv[j] = sw[n]; bv[j] = bias ? bias[n] : 0.f; }
#pragma unroll
    for (int i = 0; i < FM; i++) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
            int m = bm0 + wm0 + i * 16 + e * 2 + (lane >> 4);
            if (m < M) {
                float s = sa[m];
#pragma unroll
                for (int j = 0; j < FN; j++) {
                    int n = bn0 + wn0 + j * 16 + (lane & 15);
                    float v = (float)acc[i][j][e] * s * swv[j] + bv[j];
                    if (gelu) v = gelu_tanh(v);
                    C[(size_t)m * N + n] = f2bf(v);
                }
            }
        }
    }
}



// ---- v3: W tile through LDS (coalesced global loads, register prefetch), A fragments straight from global/L0 (one kk-step ahead):
// halves the LDS footprint (18 KB for 128x128x128) so 2-3 WGs fit per CU and barriers of one WG overlap with compute of another.
template <int BM, int BN, int BK, int WM, int WN, int MINB>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32, MINB)
i8gemm3_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ W, const float* __restrict__ sa, const float* __restrict__ sw,
               const float* __restrict__ bias, uint16_t* __restrict__ C, int M, int N, int K, int gelu, int swap, int lda, int ldw) {
    constexpr int NW = (BM / WM) * (BN / WN), NT = NW * 32;
    constexpr int LDV = BK / 16 + 1, CPR = BK / 16;
    constexpr int BPT = BN * CPR / NT;
    static_assert(BN * CPR % NT == 0, "tile/threads mismatch");
    constexpr int FM = WM / 16, FN = WN / 16;
    __shared__ v4i Bs[BN * LDV];
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int wm0 = (wid / (BN / WN)) * WM, wn0 = (wid % (BN / WN)) * WN;
    const int bm0 = (swap ? blockIdx.x : blockIdx.y) * BM, bn0 = (swap ? blockIdx.y : blockIdx.x) * BN;
    const int KT = K / BK;
    v4i rb[BPT];
    const int8_t* ap[FM];
#pragma unroll
    for (int i = 0; i < FM; i++) ap[i] = A + (size_t)min(bm0 + wm0 + i * 16 + (lane & 15), M - 1) * lda;
    auto gload = [&](int kt) {
#pragma unroll
        for (int i = 0; i < BPT; i++) { int c = tid + i * NT; int r = c / CPR, cc = c % CPR; rb[i] = *(const v4i*)(W + (size_t)(bn0 + r) * ldw + kt * BK + cc * 16); }
    };
    auto lstore = [&]() {
#pragma unroll
        for (int i = 0; i < BPT; i++) { int c = tid + i * NT; int r = c / CPR, cc = c % CPR; Bs[r * LDV + cc] = rb[i]; }
    };
    v8i acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) acc[i][j] = (v8i){0, 0, 0, 0, 0, 0, 0, 0};
    v4i af[FM], afn[FM];
#pragma unroll
    for (int i = 0; i < FM; i++) af[i] = *(const v4i*)(ap[i]);
    gload(0); lstore(); __syncthreads();
    for (int kt = 0; kt < KT; kt++) {
        if (kt + 1 < KT) gload(kt + 1);
#pragma unroll
        for (int kk = 0; kk < CPR; kk++) {
            int knext = kt * BK + (kk + 1) * 16;
            if (knext < K) {
#pragma unroll
                for (int i = 0; i < FM; i++) afn[i] = *(const v4i*)(ap[i] + knext);
            }
            v4i bf[FN];
#pragma unroll
            for (int j = 0; j < FN; j++) bf[j] = Bs[(wn0 + j * 16 + (lane & 15)) * LDV + kk];
#pragma unroll
            for (int i = 0; i < FM; i++)
#pragma unroll
                for (int j = 0; j < FN; j++) acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[i], true, bf[j], acc[i][j], false);
#pragma unroll
            for (int i = 0; i < FM; i++) af[i] = afn[i];
        }
        __syncthreads();
        if (kt + 1 < KT) { lstore(); __syncthreads(); }
    }
    float swv[FN], bv[FN];
#pragma unroll
    for (int j = 0; j < FN; j++) { int n = bn0 + wn0 + j * 16 + (lane & 15); swv[j] = sw[n]; bv[j] = bias ? bias[n] : 0.f; }
#pragma unroll
    for (int i = 0; i < FM; i++) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
            int m = bm0 + wm0 + i * 16 + e * 2 + (lane >> 4);
            if (m < M) {
                float s = sa[m];
#pragma unroll
                for (int j = 0; j < FN; j++) {
                    int n = bn0 + wn0 + j * 16 + (lane & 15);
                    float v = (float)acc[i][j][e] * s * swv[j] + bv[j];
                    if (gelu) v = gelu_tanh(v);
                    C[(size_t)m * N + n] = f2bf(v);
                }
            }
        }
    }
}

// ---- bf16 GEMM (vision encoder):  C[M,N] bf16 = A[M,K] bf16 * W[N,K]^T bf16 + bias[N] (+gelu), fp32 accumulate via WMMA
// K must be a multiple of 8 (16-byte chunks); M, N arbitrary (clamped loads, masked stores).
typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
union Frag16 { v4i q[2]; v16s s; };
template <int BM, int BN, int BK, int WM, int WN, int NBUF, int PD>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32)
bf16gemm_kernel(const uint16_t* __restrict__ A, const uint16_t* __restrict__ W, const float* __restrict__ bias, uint16_t* __restrict__ C,
                int M, int N, int K, int gelu, int swap, int lda, int ldw, int packed) {
    constexpr int NW = (BM / WM) * (BN / WN), NT = NW * 32;
    constexpr int CPR = BK / 8;                 // 16-byte chunks (8 bf16) per row
    constexpr int LDV = CPR + 1;                // padded row stride in 16-byte units
    constexpr int APT = BM * CPR / NT, BPT = BN * CPR / NT;
    static_assert(BM * CPR % NT == 0 && BN * CPR % NT == 0, "tile/threads mismatch");
    constexpr int FM = WM / 16, FN = WN / 16;
    __shared__ v4i As[NBUF][BM * LDV];
    __shared__ v4i Bs[NBUF][BN * LDV];
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int wm0 = (wid / (BN / WN)) * WM, wn0 = (wid % (BN / WN)) * WN;
    const int bm0 = (swap ? blockIdx.x : blockIdx.y) * BM, bn0 = (swap ? blockIdx.y : blockIdx.x) * BN;
    v4i ra[PD][APT], rb[PD][BPT];
    const int KT = (K + BK - 1) / BK;
    const v4i zero4 = {0, 0, 0, 0};
    auto gload = [&](int kt, auto SET) {
        constexpr int s = decltype(SET)::value;
#pragma unroll
        for (int i = 0; i < APT; i++) { int c = tid + i * NT; int r = c / CPR, cc = c % CPR; int m = min(bm0 + r, M - 1); int k = kt * BK + cc * 8; ra[s][i] = k < K ? *(const v4i*)(A + (size_t)m * lda + k) : zero4; }
#pragma unroll
        for (int i = 0; i < BPT; i++) { int c = tid + i * NT; int r = c / CPR, cc = c % CPR; int n = min(bn0 + r, N - 1); int k = kt * BK + cc * 8; rb[s][i] = packed ? *(const v4i*)(W + ((size_t)(bn0 / 128) * KT + kt) * (128 * BK) + ((bn0 % 128) + r) * BK + cc * 8) : (k < K ? *(const v4i*)(W + (size_t)n * ldw + k) : zero4); }
    };
    auto lstore = [&](auto SET, auto BUF) {
        constexpr int s = decltype(SET)::value, b = decltype(BUF)::value;
#pragma unroll
        for (int i = 0; i < APT; i++) { int c = tid + i * NT; int r = c / CPR, cc = c % CPR; As[b][r * LDV + cc] = ra[s][i]; }
#pragma unroll
        for (int i = 0; i < BPT; i++) { int c = tid + i * NT; int r = c / CPR, cc = c % CPR; Bs[b][r * LDV + cc] = rb[s][i]; }
    };
    v8f acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) acc[i][j] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
    auto compute = [&](auto BUF) {
        constexpr int b = decltype(BUF)::value;
#pragma unroll
        for (int kk = 0; kk < BK / 16; kk++) {
            Frag16 af[FM], bf[FN];
#pragma unroll
            for (int i = 0; i < FM; i++) { const v4i* p = &As[b][(wm0 + i * 16 + (lane & 15)) * LDV + kk * 2]; af[i].q[0] = p[0]; af[i].q[1] = p[1]; }
#pragma unroll
            for (int j = 0; j < FN; j++) { const v4i* p = &Bs[b][(wn0 + j * 16 + (lane & 15)) * LDV + kk * 2]; bf[j].q[0] = p[0]; bf[j].q[1] = p[1]; }
#pragma unroll
            for (int i = 0; i < FM; i++)
#pragma unroll
                for (int j = 0; j < FN; j++) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[i].s, bf[j].s, acc[i][j]);
        }
    };
    using I0 = std::integral_constant<int, 0>; using I1 = std::integral_constant<int, 1>;
    gload(0, I0{});
    if (PD > 1 && KT > 1) gload(1, I1{});
    lstore(I0{}, I0{}); __syncthreads();
    auto step = [&](int kt, auto SET, auto BUF) {
        constexpr int s = decltype(SET)::value, b = decltype(BUF)::value;
        constexpr int sload = PD == 1 ? 0 : (s ^ 1);
        constexpr int bnext = NBUF == 1 ? 0 : (b ^ 1);
        if (kt + PD < KT) gload(kt + PD, std::integral_constant<int, sload>{});
        compute(std::integral_constant<int, b>{});
        if (kt + 1 < KT) {
            if (NBUF == 1) __syncthreads();
            lstore(std::integral_constant<int, s>{}, std::integral_constant<int, bnext>{});
        }
        __syncthreads();
    };
    int kt = 0;
    for (; kt + 1 < KT; kt += 2) {
        step(kt, std::integral_constant<int, PD == 1 ? 0 : 1>{}, I0{});
        step(kt + 1, I0{}, std::integral_constant<int, NBUF == 1 ? 0 : 1>{});
    }
    if (kt < KT) step(kt, std::integral_constant<int, PD == 1 ? 0 : 1>{}, I0{});
    float bv[FN];
#pragma unroll
    for (int j = 0; j < FN; j++) { int n = bn0 + wn0 + j * 16 + (lane & 15); bv[j] = (bias && n < N) ? bias[n] : 0.f; }
#pragma unroll
    for (int i = 0; i < FM; i++) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
            int m = bm0 + wm0 + i * 16 + e * 2 + (lane >> 4);
            if (m < M) {
#pragma unroll
                for (int j = 0; j < FN; j++) {
                    int n = bn0 + wn0 + j * 16 + (lane & 15);
                    float v = acc[i][j][e] + bv[j];
                    if (gelu) v = gelu_tanh(__uint_as_float((uint32_t)f2bf(v) << 16));  // round to bf16 first: bit-identical to F.gelu(bf16 linear output)
                    if (n < N) C[(size_t)m * N + n] = f2bf(v);
                }
            }
        }
    }
}


// ---- small-M (query prefill, M <= 64) int8 GEMM: weight-streaming bound -> no LDS, direct global->register fragments, split-K
// partial int32 sums into P[splitk][M][N], then a tiny epilogue kernel (exact: same integer sums as a single pass).
template <int FM, int FN, int UNROLL>
__global__ void __launch_bounds__(128)
i8gemm_small_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ W, int32_t* __restrict__ P, int M, int N, int K, int kper, int lda, int ldw) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int n0 = blockIdx.x * (4 * 16 * FN) + wave * 16 * FN;
    const int ks = blockIdx.y, k0 = ks * kper, kend = k0 + kper;
    const int r = lane & 15;
    const int8_t* ap[FM]; const int8_t* bp[FN];
#pragma unroll
    for (int i = 0; i < FM; i++) ap[i] = A + (size_t)min(i * 16 + r, M - 1) * lda;
#pragma unroll
    for (int j = 0; j < FN; j++) bp[j] = W + (size_t)min(n0 + j * 16 + r, N - 1) * ldw;
    v8i acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) acc[i][j] = (v8i){0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = k0; k < kend; k += 16 * UNROLL) {
        v4i a[UNROLL][FM], b[UNROLL][FN];
#pragma unroll
        for (int u = 0; u < UNROLL; u++) {
#pragma unroll
            for (int j = 0; j < FN; j++) b[u][j] = *(const v4i*)(bp[j] + k + u * 16);
#pragma unroll
            for (int i = 0; i < FM; i++) a[u][i] = *(const v4i*)(ap[i] + k + u * 16);
        }
#pragma unroll
        for (int u = 0; u < UNROLL; u++)
#pragma unroll
            for (int i = 0; i < FM; i++)
#pragma unroll
                for (int j = 0; j < FN; j++) acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a[u][i], true, b[u][j], acc[i][j], false);
    }
    int32_t* Pk = P + (size_t)ks * M * N;
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
        for (int e = 0; e < 8; e++) {
            int m = i * 16 + e * 2 + (lane >> 4);
            if (m < M)
#pragma unroll
                for (int j = 0; j < FN; j++) { int n = n0 + j * 16 + r; if (n < N) Pk[(size_t)m * N + n] = acc[i][j][e]; }
        }
}
__global__ void i8gemm_small_epilogue(const int32_t* __restrict__ P, int splitk, const float* __restrict__ sa, const float* __restrict__ sw,
                                      const float* __restrict__ bias, uint16_t* __restrict__ C, int M, int N, int gelu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int m = idx / N, n = idx - m * N;
    int32_t s = 0;
    for (int k = 0; k < splitk; k++) s += P[(size_t)k * M * N + idx];
    float v = (float)s * sa[m] * sw[n] + (bias ? bias[n] : 0.f);
    if (gelu) v = gelu_tanh(v);
    C[idx] = f2bf(v);
}

// pure WMMA issue-rate probe (no memory traffic)
__global__ void wmma_peak_kernel(int iters, int* out) {
    v4i a = {(int)threadIdx.x, 1, 2, 3}, b = {4, 5, 6, (int)threadIdx.x};
    v8i c0 = {0,0,0,0,0,0,0,0}, c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0, c6 = c0, c7 = c0;
    for (int i = 0; i < iters; i++) {
        c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c0, false);
        c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c1, false);
        c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c2, false);
        c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c3, false);
        c4 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c4, false);
        c5 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c5, false);
        c6 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c6, false);
        c7 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c7, false);
    }
    int s = 0;
    for (int e = 0; e < 8; e++) s += c0[e] + c1[e] + c2[e] + c3[e] + c4[e] + c5[e] + c6[e] + c7[e];
    if (s == 0x7fffffff) out[0] = s;
}

typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
__global__ void wmma_peak_bf16_kernel(int iters, int* out) {
    v16s a, b; for (int i = 0; i < 16; i++) { a[i] = (short)(threadIdx.x + i); b[i] = (short)(i * 3); }
    v8f c0 = {0,0,0,0,0,0,0,0}, c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0, c6 = c0, c7 = c0;
    for (int i = 0; i < iters; i++) {
        c0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c0); c1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c1);
        c2 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c2); c3 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c3);
        c4 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c4); c5 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c5);
        c6 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c6); c7 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c7);
    }
    float s = 0; for (int e = 0; e < 8; e++) s += c0[e] + c1[e] + c2[e] + c3[e] + c4[e] + c5[e] + c6[e] + c7[e];
    if (s == 123456.f) out[0] = 1;
}
extern "C" {
void wmma_peak_bf16(hipStream_t st, int blocks, int threads, int iters, int* out) { wmma_peak_bf16_kernel<<<blocks, threads, 0, st>>>(iters, out); }
int i8gemm_launch(int cfg, hipStream_t st, const int8_t* A, const int8_t* W, const float* sa, const float* sw, const float* bias, uint16_t* C, int M, int N, int K, int gelu, int swap, int lda, int ldw, int packed) {
    if (packed && cfg < 10) return -4;
#define LAUNCH(BM, BN, BK, WM, WN) { if (N % BN || K % BK) return -2; dim3 g(N / BN, (M + BM - 1) / BM); if (swap) g = dim3((M + BM - 1) / BM, N / BN); i8gemm_kernel<BM, BN, BK, WM, WN><<<g, (BM / WM) * (BN / WN) * 32, 0, st>>>(A, W, sa, sw, bias, C, M, N, K, gelu, swap, lda, ldw); return 0; }
    switch (cfg) {
        case 0: LAUNCH(128, 128, 64, 64, 64)
        case 1: LAUNCH(128, 256, 64, 64, 64)
        case 2: LAUNCH(128, 128, 64, 32, 64)
        case 3: LAUNCH(64, 128, 64, 32, 64)
        case 4: LAUNCH(128, 128, 32, 64, 64)
        case 5: LAUNCH(128, 128, 32, 32, 64)
        case 6: LAUNCH(64, 64, 64, 32, 32)
        case 7: LAUNCH(256, 128, 64, 64, 64)
        case 8: LAUNCH(64, 128, 32, 32, 64)
        case 9: LAUNCH(128, 64, 64, 64, 32)
#define LAUNCH2(BM, BN, BK, WM, WN, NBUF, PD, PRIO) { if (packed && (BK != 128 || 128 % BN)) return -4; if (N % BN || K % BK) return -2; dim3 g(N / BN, (M + BM - 1) / BM); if (swap) g = dim3((M + BM - 1) / BM, N / BN); i8gemm2_kernel<BM, BN, BK, WM, WN, NBUF, PD, PRIO><<<g, (BM / WM) * (BN / WN) * 32, 0, st>>>(A, W, sa, sw, bias, C, M, N, K, gelu, swap, nullptr, K, lda, ldw, packed); return 0; }
#define LAUNCH2B(BM, BN, BK, WM, WN, NBUF, PD, PRIO, MINB) { if (packed && (BK != 128 || 128 % BN)) return -4; if (N % BN || K % BK) return -2; dim3 g(N / BN, (M + BM - 1) / BM); if (swap) g = dim3((M + BM - 1) / BM, N / BN); i8gemm2_kernel<BM, BN, BK, WM, WN, NBUF, PD, PRIO, MINB><<<g, (BM / WM) * (BN / WN) * 32, 0, st>>>(A, W, sa, sw, bias, C, M, N, K, gelu, swap, nullptr, K, lda, ldw, packed); return 0; }
        case 10: LAUNCH2(128, 128, 64, 32, 64, 2, 1, 0)
        case 11: LAUNCH2(128, 128, 64, 32, 64, 2, 2, 0)
        case 12: LAUNCH2(128, 128, 64, 32, 64, 1, 1, 0)
        case 13: LAUNCH2(128, 128, 64, 32, 64, 1, 2, 0)
        case 14: LAUNCH2(128, 256, 64, 64, 64, 1, 2, 0)
        case 15: LAUNCH2(64, 128, 64, 32, 64, 1, 2, 0)
        case 16: LAUNCH2(128, 128, 64, 64, 64, 1, 2, 0)
        case 17: LAUNCH2(128, 128, 128, 32, 64, 1, 1, 0)
        case 18: LAUNCH2(256, 128, 64, 64, 64, 1, 2, 0)
        case 19: LAUNCH2(128, 128, 64, 32, 64, 1, 2, 1)
        case 20: LAUNCH2(128, 128, 64, 32, 64, 2, 2, 1)
        case 21: LAUNCH2(64, 128, 64, 32, 64, 2, 2, 0)
        case 22: LAUNCH2(128, 64, 64, 64, 32, 1, 2, 0)
        case 23: LAUNCH2(128, 128, 128, 32, 64, 1, 2, 0)
        case 24: LAUNCH2B(128, 128, 64, 32, 64, 1, 1, 0, 2)
        case 25: LAUNCH2B(128, 128, 64, 32, 64, 2, 1, 0, 2)
        case 26: LAUNCH2B(64, 128, 64, 32, 64, 1, 1, 0, 3)
        case 27: LAUNCH2B(128, 128, 128, 32, 64, 1, 1, 0, 2)
#define LAUNCH3(BM, BN, BK, WM, WN, MINB) { if (packed) return -4; if (N % BN || K % BK) return -2; dim3 g(N / BN, (M + BM - 1) / BM); if (swap) g = dim3((M + BM - 1) / BM, N / BN); i8gemm3_kernel<BM, BN, BK, WM, WN, MINB><<<g, (BM / WM) * (BN / WN) * 32, 0, st>>>(A, W, sa, sw, bias, C, M, N, K, gelu, swap, lda, ldw); return 0; }
        case 28: LAUNCH3(128, 128, 128, 32, 64, 2)
        case 29: LAUNCH3(128, 128, 128, 32, 64, 1)
        case 30: LAUNCH3(128, 128, 64, 32, 64, 2)
        case 31: LAUNCH3(128, 128, 256, 32, 64, 1)
        case 32: LAUNCH3(128, 256, 128, 32, 64, 1)
        case 33: LAUNCH3(64, 128, 128, 32, 64, 2)
        case 34: LAUNCH3(128, 128, 128, 64, 64, 2)
        case 35: LAUNCH2B(128, 64, 128, 32, 32, 1, 1, 0, 2)
        case 36: LAUNCH2B(128, 64, 128, 64, 32, 1, 1, 0, 2)
        case 37: LAUNCH2B(64, 128, 128, 32, 64, 1, 1, 0, 2)
        case 38: LAUNCH2B(64, 128, 128, 32, 32, 1, 1, 0, 2)
        case 39: LAUNCH2(128, 128, 128, 32, 32, 1, 1, 0)
        case 40: LAUNCH2(128, 128, 64, 32, 32, 2, 1, 0)
        case 41: LAUNCH2(128, 128, 128, 16, 64, 1, 1, 0)
        default: return -1;
    }
#undef LAUNCH
}
int i8gemm_ncfg() { return 42; }
int bf16gemm_launch(int cfg, hipStream_t st, const uint16_t* A, const uint16_t* W, const float* bias, uint16_t* C, int M, int N, int K, int gelu, int swap, int lda, int ldw, int packed) {
#define LAUNCHB(BM, BN, BK, WM, WN, NBUF, PD) { if (packed && (BK != 64 || 128 % BN)) return -4; if (K % 8) return -2; dim3 g((N + BN - 1) / BN, (M + BM - 1) / BM); if (swap) g = dim3((M + BM - 1) / BM, (N + BN - 1) / BN); bf16gemm_kernel<BM, BN, BK, WM, WN, NBUF, PD><<<g, (BM / WM) * (BN / WN) * 32, 0, st>>>(A, W, bias, C, M, N, K, gelu, swap, lda, ldw, packed); return 0; }
    switch (cfg) {
        case 0: LAUNCHB(128, 128, 64, 32, 64, 1, 1)
        case 1: LAUNCHB(128, 128, 64, 32, 64, 1, 2)
        case 2: LAUNCHB(128, 128, 32, 32, 64, 2, 1)
        case 3: LAUNCHB(128, 128, 32, 32, 64, 2, 2)
        case 4: LAUNCHB(128, 128, 64, 64, 64, 1, 1)
        case 5: LAUNCHB(64, 128, 64, 32, 64, 1, 1)
        case 6: LAUNCHB(64, 128, 64, 32, 64, 2, 1)
        case 7: LAUNCHB(128, 64, 64, 64, 32, 1, 1)
        case 8: LAUNCHB(64, 64, 64, 32, 32, 2, 1)
        case 9: LAUNCHB(128, 128, 96, 32, 64, 1, 1)
        case 10: LAUNCHB(128, 64, 64, 32, 32, 1, 1)
        case 11: LAUNCHB(64, 128, 64, 32, 32, 1, 1)
        default: return -1;
    }
#undef LAUNCHB
}
int bf16gemm_ncfg() { return 12; }
// cfg: 0..2 select (FN, UNROLL); FM follows M. P must hold splitk*M*N int32 (<= 16*64*8192*4 bytes).
int i8gemm_small_launch(int cfg, hipStream_t st, const int8_t* A, const int8_t* W, int32_t* P, const float* sa, const float* sw, const float* bias, uint16_t* C, int M, int N, int K, int gelu, int lda, int ldw, int packed) {
    if (packed && (cfg < 3 || cfg == 5 || cfg == 6 || cfg == 7)) return -4;
    if (M > 64 || K % 64) return -2;
    if (cfg >= 3) {  // LDS-staged WMMA tiles with split-K (i8gemm2_kernel<..., SPLIT=true>)
        int BM = M <= 32 ? 32 : 64, BN, BK;
        if (cfg == 3) { BN = 128; BK = 128; } else if (cfg == 4) { BN = 64; BK = 128; } else if (cfg == 5) { BN = 128; BK = 64; } else if (cfg == 6) { BN = 64; BK = 256; } else { BN = 32; BK = 256; }
        if (N % BN || K % BK) return -2;
        int nwg = N / BN; int splitk = 1;
        static int split_min = getenv("I8_SPLIT_MIN") ? atoi(getenv("I8_SPLIT_MIN")) : 48;   // split K only while the grid is smaller than this
        while (nwg * splitk < split_min && splitk < 16 && (K / (splitk * 2)) % BK == 0) splitk *= 2;
        int kper = K / splitk;
        dim3 g(nwg, 1, splitk);
        // splitk == 1: plain kernel with the fused epilogue (no int32 partial round trip, no second launch)
#define LSK(BM_, BN_, BK_, WM_, WN_) { if (splitk == 1) i8gemm2_kernel<BM_, BN_, BK_, WM_, WN_, 1, 1, 0, 1, false><<<g, (BM_ / WM_) * (BN_ / WN_) * 32, 0, st>>>(A, W, sa, sw, bias, C, M, N, K, gelu, 0, nullptr, K, lda, ldw, packed); \
                                     else i8gemm2_kernel<BM_, BN_, BK_, WM_, WN_, 1, 1, 0, 1, true><<<g, (BM_ / WM_) * (BN_ / WN_) * 32, 0, st>>>(A, W, sa, sw, bias, C, M, N, K, gelu, 0, P, kper, lda, ldw, packed); }
        if (BM == 32) {
            if (cfg == 3) LSK(32, 128, 128, 32, 32) else if (cfg == 4) LSK(32, 64, 128, 32, 16) else if (cfg == 5) LSK(32, 128, 64, 32, 32) else if (cfg == 6) LSK(32, 64, 256, 32, 16) else LSK(32, 32, 256, 32, 16)
        } else {
            if (cfg == 3) LSK(64, 128, 128, 32, 64) else if (cfg == 4) LSK(64, 64, 128, 32, 32) else if (cfg == 5) LSK(64, 128, 64, 32, 64) else if (cfg == 6) LSK(64, 64, 256, 32, 32) else LSK(64, 32, 256, 32, 16)
        }
#undef LSK
        if (splitk > 1) { int tot = M * N; i8gemm_small_epilogue<<<(tot + 255) / 256, 256, 0, st>>>(P, splitk, sa, sw, bias, C, M, N, gelu); }
        return 0;
    }
    int FN = cfg == 0 ? 2 : (cfg == 1 ? 1 : 4), UNROLL = cfg == 0 ? 4 : (cfg == 1 ? 8 : 2);
    int BN = 4 * 16 * FN; int nwg = (N + BN - 1) / BN;
    int splitk = 1;
    while (nwg * splitk < 120 && splitk < 16 && (K / (splitk * 2)) % (16 * UNROLL) == 0 && K / (splitk * 2) >= 128) splitk *= 2;
    int kper = K / splitk;
    if (kper % (16 * UNROLL)) return -3;
    dim3 g(nwg, splitk);
#define LS(FM_, FN_, U_) i8gemm_small_kernel<FM_, FN_, U_><<<g, 128, 0, st>>>(A, W, P, M, N, K, kper, lda, ldw)
    if (M <= 32) { if (cfg == 0) LS(2, 2, 4); else if (cfg == 1) LS(2, 1, 8); else LS(2, 4, 2); }
    else         { if (cfg == 0) LS(4, 2, 4); else if (cfg == 1) LS(4, 1, 8); else LS(4, 4, 2); }
#undef LS
    int tot = M * N; i8gemm_small_epilogue<<<(tot + 255) / 256, 256, 0, st>>>(P, splitk, sa, sw, bias, C, M, N, gelu);
    return 0;
}
int i8gemm_small_ncfg() { return 8; }
void wmma_peak(hipStream_t st, int blocks, int threads, int iters, int* out) { wmma_peak_kernel<<<blocks, threads, 0, st>>>(iters, out); }
}
