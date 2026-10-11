// Non-causal, mask-free flash attention (prefill) for gfx11 wave32 WMMA bf16: O = softmax(Q K^T * scale) V
// Q: (H, T, D) with strides (sq_h, sq_t); K: (H, T, D) strides (sk_h, sk_t); VT: (H, DP, TP) contiguous, V transposed per head,
// zero padded to TP = ceil(T/64)*64 keys and DP >= D dims; O: (T, H*D) row stride so_t, head offset h*D.  D in {64, 72}; DP = 64 or 80.
#include <hip/hip_runtime.h>
#include <stdint.h>
typedef int v4i __attribute__((ext_vector_type(4)));
typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
union Frag16 { v4i q[2]; v16s s; };
__device__ __forceinline__ uint16_t f2bf(float f) { uint32_t u = __float_as_uint(f); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
__device__ __forceinline__ float shfl_xor(float v, int m) { return __shfl_xor(v, m, 32); }

template <int DP>   // padded head dim (multiple of 16): 64 or 80
__global__ void __launch_bounds__(128)
attn_fwd_kernel(const uint16_t* __restrict__ Q, const uint16_t* __restrict__ K, const uint16_t* __restrict__ VT, uint16_t* __restrict__ O,
                int T, int TP, int D, long sq_h, long sq_t, long sk_h, long sk_t, long so_t, float scale) {
    constexpr int BK = 64;                 // keys per block
    constexpr int KS = DP / 8 + 1;         // K tile row stride in 16-B chunks (DP*2+16 bytes): conflict-free
    constexpr int VS = BK / 8 + 1;         // VT tile row stride in chunks (144 B)
    constexpr int PS = BK / 8 + 1;         // P tile row stride
    constexpr int DF = DP / 16;            // d fragments
    constexpr int KCH = DP / 8;            // chunks per K row
    __shared__ v4i Ks[BK * KS];            // 64 x DP bf16
    __shared__ v4i Vs[DP * VS];            // DP x 64 bf16 (transposed V)
    __shared__ v4i Ps[4][16 * PS];         // per wave: 16 x 64 bf16 probabilities
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, r = lane & 15, hi = lane >> 4;
    const int h = blockIdx.y, q0 = blockIdx.x * 64 + wave * 16;
    const uint16_t* Qh = Q + h * sq_h; const uint16_t* Kh = K + h * sk_h; const uint16_t* VTh = VT + (size_t)h * DP * TP;
    // Q fragments (A operand): lane holds query row q0 + r, 16 consecutive dims per fragment
    Frag16 qf[DF];
    {
        int qr = min(q0 + r, T - 1);
        const uint16_t* qp = Qh + (size_t)qr * sq_t;
#pragma unroll
        for (int f = 0; f < DF; f++) {
            if (f * 16 + 8 <= D) { qf[f].q[0] = *(const v4i*)(qp + f * 16); } else { qf[f].q[0] = (v4i){0, 0, 0, 0}; }
            if (f * 16 + 16 <= D) { qf[f].q[1] = *(const v4i*)(qp + f * 16 + 8); } else { qf[f].q[1] = (v4i){0, 0, 0, 0}; }
        }
    }
    v8f o[DF];
#pragma unroll
    for (int f = 0; f < DF; f++) o[f] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
    float m_run[8], l_run[8];
#pragma unroll
    for (int e = 0; e < 8; e++) { m_run[e] = -1e30f; l_run[e] = 0.f; }
    const int NB = TP / BK;
    // cooperative tile loads: K tile BK rows x KCH chunks; VT tile DP rows x 8 chunks
    constexpr int KCHUNKS = BK * KCH, VCHUNKS = DP * 8;
    constexpr int KPT = (KCHUNKS + 127) / 128, VPT = (VCHUNKS + 127) / 128;
    v4i rk[KPT], rv[VPT];
    auto gload = [&](int kb) {
#pragma unroll
        for (int i = 0; i < KPT; i++) {
            int c = tid + i * 128; if (c < KCHUNKS) { int row = c / KCH, cc = c % KCH; int key = kb * BK + row;
                rk[i] = (key < T && cc * 8 + 8 <= D) ? *(const v4i*)(Kh + (size_t)key * sk_t + cc * 8) : (v4i){0, 0, 0, 0}; }
        }
#pragma unroll
        for (int i = 0; i < VPT; i++) {
            int c = tid + i * 128; if (c < VCHUNKS) { int row = c / 8, cc = c % 8;
                rv[i] = *(const v4i*)(VTh + (size_t)row * TP + kb * BK + cc * 8); }
        }
    };
    auto lstore = [&]() {
#pragma unroll
        for (int i = 0; i < KPT; i++) { int c = tid + i * 128; if (c < KCHUNKS) { int row = c / KCH, cc = c % KCH; Ks[row * KS + cc] = rk[i]; } }
#pragma unroll
        for (int i = 0; i < VPT; i++) { int c = tid + i * 128; if (c < VCHUNKS) { int row = c / 8, cc = c % 8; Vs[row * VS + cc] = rv[i]; } }
    };
    gload(0); lstore(); __syncthreads();
    for (int kb = 0; kb < NB; kb++) {
        if (kb + 1 < NB) gload(kb + 1);
        // ---- S = Q K^T  (4 key tiles of 16)
        v8f s[4];
#pragma unroll
        for (int ks = 0; ks < 4; ks++) s[ks] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int f = 0; f < DF; f++) {
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                Frag16 kf; const v4i* p = &Ks[(ks * 16 + r) * KS + f * 2]; kf.q[0] = p[0]; kf.q[1] = p[1];
                s[ks] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(qf[f].s, kf.s, s[ks]);
            }
        }
        // ---- online softmax: lane holds rows e*2+hi (e=0..7), key ks*16 + r of each tile
        float mx[8], alpha[8];
#pragma unroll
        for (int e = 0; e < 8; e++) {
            float mv = -1e30f;
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                int key = kb * BK + ks * 16 + r;
                float v = s[ks][e] * scale; if (key >= T) v = -1e30f;
                s[ks][e] = v; mv = fmaxf(mv, v);
            }
            mv = fmaxf(mv, shfl_xor(mv, 1)); mv = fmaxf(mv, shfl_xor(mv, 2)); mv = fmaxf(mv, shfl_xor(mv, 4)); mv = fmaxf(mv, shfl_xor(mv, 8));
            float mn = fmaxf(m_run[e], mv);
            alpha[e] = __expf(m_run[e] - mn); mx[e] = mn; m_run[e] = mn;
        }
        float rs[8];
#pragma unroll
        for (int e = 0; e < 8; e++) {
            float sum = 0.f;
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                float p = __expf(s[ks][e] - mx[e]);
                uint16_t pb = f2bf(p); sum += __uint_as_float((uint32_t)pb << 16);   // sum what the PV matmul will see
                ((uint16_t*)&Ps[wave][0])[(e * 2 + hi) * (PS * 8) + ks * 16 + r] = pb;
            }
            sum += shfl_xor(sum, 1); sum += shfl_xor(sum, 2); sum += shfl_xor(sum, 4); sum += shfl_xor(sum, 8);
            rs[e] = sum; l_run[e] = l_run[e] * alpha[e] + sum;
        }
#pragma unroll
        for (int f = 0; f < DF; f++)
#pragma unroll
            for (int e = 0; e < 8; e++) o[f][e] *= alpha[e];
        // ---- O += P V : P as A operand (row = query r, 16 keys), VT as B operand (col = dim, 16 keys)
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            Frag16 pf; const v4i* pp = &Ps[wave][r * PS + ks * 2]; pf.q[0] = pp[0]; pf.q[1] = pp[1];
#pragma unroll
            for (int f = 0; f < DF; f++) {
                Frag16 vf; const v4i* vp = &Vs[(f * 16 + r) * VS + ks * 2]; vf.q[0] = vp[0]; vf.q[1] = vp[1];
                o[f] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf.s, vf.s, o[f]);
            }
        }
        __syncthreads();
        if (kb + 1 < NB) { lstore(); __syncthreads(); }
    }
    // ---- epilogue: O / l, store (T, H*D)
#pragma unroll
    for (int e = 0; e < 8; e++) {
        int qr = q0 + e * 2 + hi;
        if (qr < T) {
            float inv = 1.f / l_run[e];
#pragma unroll
            for (int f = 0; f < DF; f++) { int d = f * 16 + r; if (d < D) O[(size_t)qr * so_t + h * D + d] = f2bf(o[f][e] * inv); }
        }
    }
}


// ---- v2: S^T = K Q^T formulation: the accumulator then holds 8 keys of ONE query per lane, so the row-wise softmax is in-lane
// (plus one exchange with the partner half-wave) and the P fragment for P*V is assembled in registers (no LDS round trip).
template <int DP>
__global__ void __launch_bounds__(128)
attn_fwd2_kernel(const uint16_t* __restrict__ Q, const uint16_t* __restrict__ K, const uint16_t* __restrict__ VT, uint16_t* __restrict__ O,
                 int T, int TP, int D, long sq_h, long sq_t, long sk_h, long sk_t, long so_t, float scale_log2) {
    constexpr int BK = 64, KS = DP / 8 + 1, VS = BK / 8 + 1, DF = DP / 16, KCH = DP / 8;
    __shared__ v4i Ks[BK * KS];
    __shared__ v4i Vs[DP * VS];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, r = lane & 15, hi = lane >> 4;
    const int h = blockIdx.y, q0 = blockIdx.x * 64 + wave * 16;
    const uint16_t* Qh = Q + h * sq_h; const uint16_t* Kh = K + h * sk_h; const uint16_t* VTh = VT + (size_t)h * DP * TP;
    Frag16 qf[DF];
    {
        int qr = min(q0 + r, T - 1);
        const uint16_t* qp = Qh + (size_t)qr * sq_t;
#pragma unroll
        for (int f = 0; f < DF; f++) {
            qf[f].q[0] = (f * 16 + 8 <= D) ? *(const v4i*)(qp + f * 16) : (v4i){0, 0, 0, 0};
            qf[f].q[1] = (f * 16 + 16 <= D) ? *(const v4i*)(qp + f * 16 + 8) : (v4i){0, 0, 0, 0};
        }
    }
    v8f o[DF];
#pragma unroll
    for (int f = 0; f < DF; f++) o[f] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
    float m_run = -1e30f, l_run = 0.f;   // for query q0 + r (same in both half-waves)
    const int NB = TP / BK;
    constexpr int KCHUNKS = BK * KCH, VCHUNKS = DP * 8;
    constexpr int KPT = (KCHUNKS + 127) / 128, VPT = (VCHUNKS + 127) / 128;
    v4i rk[KPT], rv[VPT];
    auto gload = [&](int kb) {
#pragma unroll
        for (int i = 0; i < KPT; i++) {
            int c = tid + i * 128; if (c < KCHUNKS) { int row = c / KCH, cc = c % KCH; int key = kb * BK + row;
                rk[i] = (key < T && cc * 8 + 8 <= D) ? *(const v4i*)(Kh + (size_t)key * sk_t + cc * 8) : (v4i){0, 0, 0, 0}; }
        }
#pragma unroll
        for (int i = 0; i < VPT; i++) {
            int c = tid + i * 128; if (c < VCHUNKS) { int row = c / 8, cc = c % 8; rv[i] = *(const v4i*)(VTh + (size_t)row * TP + kb * BK + cc * 8); }
        }
    };
    auto lstore = [&]() {
#pragma unroll
        for (int i = 0; i < KPT; i++) { int c = tid + i * 128; if (c < KCHUNKS) { int row = c / KCH, cc = c % KCH; Ks[row * KS + cc] = rk[i]; } }
#pragma unroll
        for (int i = 0; i < VPT; i++) { int c = tid + i * 128; if (c < VCHUNKS) { int row = c / 8, cc = c % 8; Vs[row * VS + cc] = rv[i]; } }
    };
    gload(0); lstore(); __syncthreads();
    for (int kb = 0; kb < NB; kb++) {
        if (kb + 1 < NB) gload(kb + 1);
        v8f s[4];
#pragma unroll
        for (int ks = 0; ks < 4; ks++) s[ks] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int f = 0; f < DF; f++) {
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                Frag16 kf; const v4i* p = &Ks[(ks * 16 + r) * KS + f * 2]; kf.q[0] = p[0]; kf.q[1] = p[1];
                s[ks] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(kf.s, qf[f].s, s[ks]);   // S^T: rows = keys, cols = queries
            }
        }
        // element e of s[ks]: key kb*64 + ks*16 + e*2 + hi, query q0 + r
        float mv = -1e30f;
        if (kb == NB - 1) {
#pragma unroll
            for (int ks = 0; ks < 4; ks++)
#pragma unroll
                for (int e = 0; e < 8; e++) { int key = kb * BK + ks * 16 + e * 2 + hi; float v = key < T ? s[ks][e] * scale_log2 : -1e30f; s[ks][e] = v; mv = fmaxf(mv, v); }
        } else {
#pragma unroll
            for (int ks = 0; ks < 4; ks++)
#pragma unroll
                for (int e = 0; e < 8; e++) { float v = s[ks][e] * scale_log2; s[ks][e] = v; mv = fmaxf(mv, v); }
        }
        mv = fmaxf(mv, shfl_xor(mv, 16));
        float mn = fmaxf(m_run, mv);
        float alpha = exp2f(m_run - mn); m_run = mn;
        float sum = 0.f;
        uint32_t pk[4][4];   // packed bf16 pairs: tile ks, dword j holds this lane's keys e=2j, 2j+1 (parity hi)
#pragma unroll
        for (int ks = 0; ks < 4; ks++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                uint16_t p0 = f2bf(exp2f(s[ks][2 * j] - mn)), p1 = f2bf(exp2f(s[ks][2 * j + 1] - mn));
                sum += __uint_as_float((uint32_t)p0 << 16) + __uint_as_float((uint32_t)p1 << 16);
                pk[ks][j] = (uint32_t)p0 | ((uint32_t)p1 << 16);
            }
        sum += shfl_xor(sum, 16);
        l_run = l_run * alpha + sum;
        // rescale O rows: O element e <-> query e*2+hi; its alpha lives in lane (e*2+hi)
        float al[8];
#pragma unroll
        for (int e = 0; e < 8; e++) al[e] = __shfl(alpha, e * 2 + hi, 32);
#pragma unroll
        for (int f = 0; f < DF; f++)
#pragma unroll
            for (int e = 0; e < 8; e++) o[f][e] *= al[e];
        // assemble P fragments: element k of the A operand (16 keys of query r) = key ks*16 + k; this lane has parity hi, the partner the other
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            Frag16 pf;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                uint32_t mine = pk[ks][j], other = __shfl_xor((int)mine, 16, 32);
                uint32_t ev = hi ? other : mine, od = hi ? mine : other;        // even keys (2j*2, (2j+1)*2) / odd keys
                // keys 4j, 4j+1, 4j+2, 4j+3 = ev.lo, od.lo, ev.hi, od.hi
                pf.q[j / 2][(j % 2) * 2] = (int)((ev & 0xffffu) | (od << 16));
                pf.q[j / 2][(j % 2) * 2 + 1] = (int)((ev >> 16) | (od & 0xffff0000u));
            }
#pragma unroll
            for (int f = 0; f < DF; f++) {
                Frag16 vf; const v4i* vp = &Vs[(f * 16 + r) * VS + ks * 2]; vf.q[0] = vp[0]; vf.q[1] = vp[1];
                o[f] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf.s, vf.s, o[f]);
            }
        }
        __syncthreads();
        if (kb + 1 < NB) { lstore(); __syncthreads(); }
    }
    float lr[8];
#pragma unroll
    for (int e = 0; e < 8; e++) lr[e] = __shfl(l_run, e * 2 + hi, 32);
#pragma unroll
    for (int e = 0; e < 8; e++) {
        int qr = q0 + e * 2 + hi;
        if (qr < T) {
            float inv = 1.f / lr[e];
#pragma unroll
            for (int f = 0; f < DF; f++) { int d = f * 16 + r; if (d < D) O[(size_t)qr * so_t + h * D + d] = f2bf(o[f][e] * inv); }
        }
    }
}


// V (H, T, D) strided -> VT (H, DP, TP) contiguous (zero padded) via a 64x64 LDS tile: coalesced reads along D, coalesced writes along T
__global__ void __launch_bounds__(256) vtranspose_kernel(const uint16_t* __restrict__ V, uint16_t* __restrict__ VT, int T, int TP, int D, int DP, long sv_h, long sv_t) {
    __shared__ uint16_t tile[64][66];
    const int h = blockIdx.z, t0 = blockIdx.x * 64, d0 = blockIdx.y * 64, tid = threadIdx.x;
    const uint16_t* Vh = V + h * sv_h; uint16_t* VTh = VT + (size_t)h * DP * TP;
    for (int i = tid; i < 64 * 64; i += 256) { int tt = i / 64, dd = i % 64; int t = t0 + tt, d = d0 + dd; tile[tt][dd] = (t < T && d < D) ? Vh[(size_t)t * sv_t + d] : 0; }
    __syncthreads();
    for (int i = tid; i < 64 * 64; i += 256) { int dd = i / 64, tt = i % 64; int t = t0 + tt, d = d0 + dd; if (d < DP) VTh[(size_t)d * TP + t] = tile[tt][dd]; }
}
extern "C" int vtranspose(hipStream_t st, const uint16_t* V, uint16_t* VT, int T, int TP, int H, int D, int DP, long sv_h, long sv_t) {
    dim3 g(TP / 64, (DP + 63) / 64, H);
    vtranspose_kernel<<<g, 256, 0, st>>>(V, VT, T, TP, D, DP, sv_h, sv_t);
    return 0;
}


// ---- v3: like v2 but each wave owns 32 query rows (two query tiles): K/V fragments from LDS are reused twice
template <int DP>
__global__ void __launch_bounds__(128)
attn_fwd3_kernel(const uint16_t* __restrict__ Q, const uint16_t* __restrict__ K, const uint16_t* __restrict__ VT, uint16_t* __restrict__ O,
                 int T, int TP, int D, long sq_h, long sq_t, long sk_h, long sk_t, long so_t, float scale_log2) {
    constexpr int BK = 64, KS = DP / 8 + 1, VS = BK / 8 + 1, DF = DP / 16, KCH = DP / 8;
    __shared__ v4i Ks[BK * KS];
    __shared__ v4i Vs[DP * VS];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, r = lane & 15, hi = lane >> 4;
    const int h = blockIdx.y, q0 = blockIdx.x * 128 + wave * 32;
    const uint16_t* Qh = Q + h * sq_h; const uint16_t* Kh = K + h * sk_h; const uint16_t* VTh = VT + (size_t)h * DP * TP;
    Frag16 qf[2][DF];
#pragma unroll
    for (int t = 0; t < 2; t++) {
        int qr = min(q0 + t * 16 + r, T - 1);
        const uint16_t* qp = Qh + (size_t)qr * sq_t;
#pragma unroll
        for (int f = 0; f < DF; f++) {
            qf[t][f].q[0] = (f * 16 + 8 <= D) ? *(const v4i*)(qp + f * 16) : (v4i){0, 0, 0, 0};
            qf[t][f].q[1] = (f * 16 + 16 <= D) ? *(const v4i*)(qp + f * 16 + 8) : (v4i){0, 0, 0, 0};
        }
    }
    v8f o[2][DF];
#pragma unroll
    for (int t = 0; t < 2; t++)
#pragma unroll
        for (int f = 0; f < DF; f++) o[t][f] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
    float m_run[2] = {-1e30f, -1e30f}, l_run[2] = {0.f, 0.f};
    const int NB = TP / BK;
    constexpr int KCHUNKS = BK * KCH, VCHUNKS = DP * 8;
    constexpr int KPT = (KCHUNKS + 127) / 128, VPT = (VCHUNKS + 127) / 128;
    v4i rk[KPT], rv[VPT];
    auto gload = [&](int kb) {
#pragma unroll
        for (int i = 0; i < KPT; i++) {
            int c = tid + i * 128; if (c < KCHUNKS) { int row = c / KCH, cc = c % KCH; int key = kb * BK + row;
                rk[i] = (key < T && cc * 8 + 8 <= D) ? *(const v4i*)(Kh + (size_t)key * sk_t + cc * 8) : (v4i){0, 0, 0, 0}; }
        }
#pragma unroll
        for (int i = 0; i < VPT; i++) {
            int c = tid + i * 128; if (c < VCHUNKS) { int row = c / 8, cc = c % 8; rv[i] = *(const v4i*)(VTh + (size_t)row * TP + kb * BK + cc * 8); }
        }
    };
    auto lstore = [&]() {
#pragma unroll
        for (int i = 0; i < KPT; i++) { int c = tid + i * 128; if (c < KCHUNKS) { int row = c / KCH, cc = c % KCH; Ks[row * KS + cc] = rk[i]; } }
#pragma unroll
        for (int i = 0; i < VPT; i++) { int c = tid + i * 128; if (c < VCHUNKS) { int row = c / 8, cc = c % 8; Vs[row * VS + cc] = rv[i]; } }
    };
    gload(0); lstore(); __syncthreads();
    for (int kb = 0; kb < NB; kb++) {
        if (kb + 1 < NB) gload(kb + 1);
        v8f s[2][4];
#pragma unroll
        for (int t = 0; t < 2; t++)
#pragma unroll
            for (int ks = 0; ks < 4; ks++) s[t][ks] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int f = 0; f < DF; f++) {
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                Frag16 kf; const v4i* p = &Ks[(ks * 16 + r) * KS + f * 2]; kf.q[0] = p[0]; kf.q[1] = p[1];
                s[0][ks] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(kf.s, qf[0][f].s, s[0][ks]);
                s[1][ks] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(kf.s, qf[1][f].s, s[1][ks]);
            }
        }
        uint32_t pk[2][4][4];
        float al[2][8];
#pragma unroll
        for (int t = 0; t < 2; t++) {
            float mv = -1e30f;
            if (kb == NB - 1) {
#pragma unroll
                for (int ks = 0; ks < 4; ks++)
#pragma unroll
                    for (int e = 0; e < 8; e++) { int key = kb * BK + ks * 16 + e * 2 + hi; float v = key < T ? s[t][ks][e] * scale_log2 : -1e30f; s[t][ks][e] = v; mv = fmaxf(mv, v); }
            } else {
#pragma unroll
                for (int ks = 0; ks < 4; ks++)
#pragma unroll
                    for (int e = 0; e < 8; e++) { float v = s[t][ks][e] * scale_log2; s[t][ks][e] = v; mv = fmaxf(mv, v); }
            }
            mv = fmaxf(mv, shfl_xor(mv, 16));
            float mn = fmaxf(m_run[t], mv);
            float alpha = exp2f(m_run[t] - mn); m_run[t] = mn;
            float sum = 0.f;
#pragma unroll
            for (int ks = 0; ks < 4; ks++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    uint16_t p0 = f2bf(exp2f(s[t][ks][2 * j] - mn)), p1 = f2bf(exp2f(s[t][ks][2 * j + 1] - mn));
                    sum += __uint_as_float((uint32_t)p0 << 16) + __uint_as_float((uint32_t)p1 << 16);
                    pk[t][ks][j] = (uint32_t)p0 | ((uint32_t)p1 << 16);
                }
            sum += shfl_xor(sum, 16);
            l_run[t] = l_run[t] * alpha + sum;
#pragma unroll
            for (int e = 0; e < 8; e++) al[t][e] = __shfl(alpha, e * 2 + hi, 32);
#pragma unroll
            for (int f = 0; f < DF; f++)
#pragma unroll
                for (int e = 0; e < 8; e++) o[t][f][e] *= al[t][e];
        }
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            Frag16 pf[2];
#pragma unroll
            for (int t = 0; t < 2; t++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    uint32_t mine = pk[t][ks][j], other = __shfl_xor((int)mine, 16, 32);
                    uint32_t ev = hi ? other : mine, od = hi ? mine : other;
                    pf[t].q[j / 2][(j % 2) * 2] = (int)((ev & 0xffffu) | (od << 16));
                    pf[t].q[j / 2][(j % 2) * 2 + 1] = (int)((ev >> 16) | (od & 0xffff0000u));
                }
#pragma unroll
            for (int f = 0; f < DF; f++) {
                Frag16 vf; const v4i* vp = &Vs[(f * 16 + r) * VS + ks * 2]; vf.q[0] = vp[0]; vf.q[1] = vp[1];
                o[0][f] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf[0].s, vf.s, o[0][f]);
                o[1][f] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf[1].s, vf.s, o[1][f]);
            }
        }
        __syncthreads();
        if (kb + 1 < NB) { lstore(); __syncthreads(); }
    }
#pragma unroll
    for (int t = 0; t < 2; t++) {
        float lr[8];
#pragma unroll
        for (int e = 0; e < 8; e++) lr[e] = __shfl(l_run[t], e * 2 + hi, 32);
#pragma unroll
        for (int e = 0; e < 8; e++) {
            int qr = q0 + t * 16 + e * 2 + hi;
            if (qr < T) {
                float inv = 1.f / lr[e];
#pragma unroll
                for (int f = 0; f < DF; f++) { int d = f * 16 + r; if (d < D) O[(size_t)qr * so_t + h * D + d] = f2bf(o[t][f][e] * inv); }
            }
        }
    }
}

extern "C" int attn_fwd(hipStream_t st, const uint16_t* Q, const uint16_t* K, const uint16_t* VT, uint16_t* O, int T, int TP, int H, int D, int DP,
                        long sq_h, long sq_t, long sk_h, long sk_t, long so_t, float scale) {
    if (TP % 64 || T > TP) return -2;
    dim3 g((T + 63) / 64, H);
    if (DP == 64) attn_fwd_kernel<64><<<g, 128, 0, st>>>(Q, K, VT, O, T, TP, D, sq_h, sq_t, sk_h, sk_t, so_t, scale);
    else if (DP == 80) attn_fwd_kernel<80><<<g, 128, 0, st>>>(Q, K, VT, O, T, TP, D, sq_h, sq_t, sk_h, sk_t, so_t, scale);
    else return -1;
    return 0;
}
extern "C" int attn_fwd2(hipStream_t st, const uint16_t* Q, const uint16_t* K, const uint16_t* VT, uint16_t* O, int T, int TP, int H, int D, int DP,
                         long sq_h, long sq_t, long sk_h, long sk_t, long so_t, float scale) {
    if (TP % 64 || T > TP) return -2;
    dim3 g((T + 63) / 64, H);
    float sl2 = scale * 1.4426950408889634f;
    if (DP == 64) attn_fwd2_kernel<64><<<g, 128, 0, st>>>(Q, K, VT, O, T, TP, D, sq_h, sq_t, sk_h, sk_t, so_t, sl2);
    else if (DP == 80) attn_fwd2_kernel<80><<<g, 128, 0, st>>>(Q, K, VT, O, T, TP, D, sq_h, sq_t, sk_h, sk_t, so_t, sl2);
    else return -1;
    return 0;
}

extern "C" int attn_fwd3(hipStream_t st, const uint16_t* Q, const uint16_t* K, const uint16_t* VT, uint16_t* O, int T, int TP, int H, int D, int DP,
                         long sq_h, long sq_t, long sk_h, long sk_t, long so_t, float scale) {
    if (TP % 64 || T > TP) return -2;
    dim3 g((T + 127) / 128, H);
    float sl2 = scale * 1.4426950408889634f;
    if (DP == 64) attn_fwd3_kernel<64><<<g, 128, 0, st>>>(Q, K, VT, O, T, TP, D, sq_h, sq_t, sk_h, sk_t, so_t, sl2);
    else if (DP == 80) attn_fwd3_kernel<80><<<g, 128, 0, st>>>(Q, K, VT, O, T, TP, D, sq_h, sq_t, sk_h, sk_t, so_t, sl2);
    else return -1;
    return 0;
}
