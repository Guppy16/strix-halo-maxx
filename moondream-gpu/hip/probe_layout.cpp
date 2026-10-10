// Determine the lane/element layout of the gfx11 wave32 int8 WMMA (A, B operands and C accumulator).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
// candidate: lane l holds A[l%16][k=0..15] and B[k=0..15][l%16] (16 contiguous int8 each)
__global__ void k(const int8_t* A, const int8_t* Bt, int* out) {
    int l = threadIdx.x;
    v4i a, b;
    const int* ap = (const int*)(A + (l % 16) * 16);
    const int* bp = (const int*)(Bt + (l % 16) * 16);   // Bt is B transposed: Bt[n][k]
    for (int i = 0; i < 4; i++) { a[i] = ap[i]; b[i] = bp[i]; }
    v8i c = {0, 0, 0, 0, 0, 0, 0, 0};
    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
    for (int e = 0; e < 8; e++) out[l * 8 + e] = c[e];
}
int main() {
    std::vector<int8_t> A(256), Bt(256);
    for (int i = 0; i < 256; i++) { A[i] = (int8_t)(rand() % 21 - 10); Bt[i] = (int8_t)(rand() % 21 - 10); }
    int ref[16][16];
    for (int m = 0; m < 16; m++) for (int n = 0; n < 16; n++) { int s = 0; for (int kk = 0; kk < 16; kk++) s += (int)A[m * 16 + kk] * (int)Bt[n * 16 + kk]; ref[m][n] = s; }
    int8_t *dA, *dB; int* dO; hipMalloc(&dA, 256); hipMalloc(&dB, 256); hipMalloc(&dO, 32 * 8 * 4);
    hipMemcpy(dA, A.data(), 256, hipMemcpyHostToDevice); hipMemcpy(dB, Bt.data(), 256, hipMemcpyHostToDevice);
    k<<<1, 32>>>(dA, dB, dO); std::vector<int> out(256); hipMemcpy(out.data(), dO, 1024, hipMemcpyDeviceToHost);
    // for each (lane, ele) find which (m,n) it equals (values are distinct enough with random data)
    int ok1 = 0, ok2 = 0;
    for (int l = 0; l < 32; l++) for (int e = 0; e < 8; e++) {
        int v = out[l * 8 + e];
        int r1 = e * 2 + l / 16, c1 = l % 16;         // candidate 1
        int r2 = e + 8 * (l / 16), c2 = l % 16;       // candidate 2
        if (ref[r1][c1] == v) ok1++; if (ref[r2][c2] == v) ok2++;
    }
    printf("candidate1 (row=e*2+l/16, col=l%%16): %d/256 match; candidate2 (row=e+8*(l/16)): %d/256 match\n", ok1, ok2);
    if (ok1 != 256 && ok2 != 256) { // brute force: print mapping for lane 0..1
        for (int l = 0; l < 32; l += 16) for (int e = 0; e < 8; e++) { int v = out[l * 8 + e]; for (int m = 0; m < 16; m++) for (int n = 0; n < 16; n++) if (ref[m][n] == v) printf("lane %d ele %d -> (%d,%d)\n", l, e, m, n); }
    }
    return 0;
}
