// round 10: GELU-by-table loop variants (bundle counts only): python3 build/asm/loopcount.py on the asm
#include "../mm8.cc"
extern "C" {
// v1: separate restrict in / out, one vector per trip
__attribute__((noinline)) void gelu_v1(const bfloat16 *__restrict in, bfloat16 *__restrict out, int n, const float *ab, const float *cd) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int i = 0; i < n; i++) {
    aie::store_v(out, gelu_lut_i(aie::load_v<32>(in), ab, cd));
    in += 32;
    out += 32;
  }
}
// v2: two vectors per trip, restrict
__attribute__((noinline)) void gelu_v2(const bfloat16 *__restrict in, bfloat16 *__restrict out, int n, const float *ab, const float *cd) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int i = 0; i < n; i += 2) {
    B32 a = gelu_lut_i(aie::load_v<32>(in), ab, cd), b = gelu_lut_i(aie::load_v<32>(in + 32), ab, cd);
    aie::store_v(out, a);
    aie::store_v(out + 32, b);
    in += 64;
    out += 64;
  }
}
// v3: constant trip count 8 (one 1 KB half piece)
__attribute__((noinline)) void gelu_v3(const bfloat16 *__restrict in, bfloat16 *__restrict out, const float *ab, const float *cd) {
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < 8; i++) {
    aie::store_v(out, gelu_lut_i(aie::load_v<32>(in), ab, cd));
    in += 32;
    out += 32;
  }
}
}

extern "C" __attribute__((noinline)) void gelu_ii16(const bfloat16 *__restrict in, bfloat16 *__restrict out, int n, const float *ab, const float *cd) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
#pragma clang loop pipeline_initiation_interval(16)
  for (int i = 0; i < n; i++) {
    aie::store_v(out, gelu_lut_i(aie::load_v<32>(in), ab, cd));
    in += 32;
    out += 32;
  }
}

extern "C" __attribute__((noinline)) void gelu_ii20(const bfloat16 *__restrict in, bfloat16 *__restrict out, int n, const float *ab, const float *cd) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
#pragma clang loop pipeline_initiation_interval(20)
  for (int i = 0; i < n; i++) {
    aie::store_v(out, gelu_lut_i(aie::load_v<32>(in), ab, cd));
    in += 32;
    out += 32;
  }
}

extern "C" __attribute__((noinline)) void gelu_ii24(const bfloat16 *__restrict in, bfloat16 *__restrict out, int n, const float *ab, const float *cd) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
#pragma clang loop pipeline_initiation_interval(24)
  for (int i = 0; i < n; i++) {
    aie::store_v(out, gelu_lut_i(aie::load_v<32>(in), ab, cd));
    in += 32;
    out += 32;
  }
}
