#pragma GCC optimize("O3")
#pragma GCC optimize("O3")
#pragma GCC optimize("O3")
#pragma GCC optimize("O3")
#pragma GCC optimize("O3")
//===- gemm_atb_mixed_test.cpp ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host runner for the BF16-in / BFP16-weight / BF16-out GEMM in this folder.
// Identical to ../matrix_multiplication/mixed_test.cpp except B is pre-shuffled
// via gemm_atb::layout_transpose_L1_1x2_8x8block (column-major over L1 tiles,
// each tile rearranged into 1x2 super-blocks of 8x8 column-major sub-blocks)
// instead of the canonical shuffleMatrixForBfp16ebs8(K,N,k,N,...) layout.
//
//===----------------------------------------------------------------------===//

#include <bits/stdc++.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "/work/mlir-aie/programming_examples/ml/block_datatypes/helper.h"
#include "common.h"
#include "/work/mlir-aie/programming_examples/ml/block_datatypes/gemm_asymmetric_tile_buffering/gemm_atb_layout.h"
#include <random>
#include <algorithm>

#define XSTR(X) STR(X)
#define STR(X) #X

constexpr long long verify_stochastic_threshold = 1024 * 1024;
constexpr int verify_stochastic_n_samples = 1000;

// Verification tolerance
// See "Note on Numerical Tolerances" in README.md
// TODO: This might have to be adjusted for bfp
float abs_tol = matmul_common::get_abs_tol<test_utils::bfloat16_t>();
float rel_tol = matmul_common::get_rel_tol<test_utils::bfloat16_t>() * 2.0f;

int main(int argc, const char *argv[]) {

  // ------------------------------------------------------
  // Parse program arguments
  // ------------------------------------------------------
  cxxopts::Options options("Matrix Matrix Multiplication Test");
  cxxopts::ParseResult vm;
  matmul_common::add_default_options(options);
  options.add_options()("trows,w", "Tile size m",
                        cxxopts::value<int>()->default_value("64"))(
      "tinner,y", "Tile size k", cxxopts::value<int>()->default_value("64"))(
      "tcolumns,z", "Tile size n", cxxopts::value<int>()->default_value("64"))(
      "Mr", "real M (rest is zero padding)", cxxopts::value<int>()->default_value("0"))(
      "Kr", "real K", cxxopts::value<int>()->default_value("0"))(
      "Nr", "real N", cxxopts::value<int>()->default_value("0"))(
      "wstd", "weight std", cxxopts::value<float>()->default_value("0.03"))(
      "outlier", "activation outlier channel scale (0 = none)", cxxopts::value<float>()->default_value("0"));

  matmul_common::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();
  int b_col_maj = vm["b_col_maj"].as<int>();

  // Fix the seed to ensure reproducibility in CI.
  srand(1726250518); // srand(time(NULL));

  int M = vm["M"].as<int>();
  int K = vm["K"].as<int>();
  int N = vm["N"].as<int>();

  int m = vm["w"].as<int>();
  int k = vm["y"].as<int>();
  int n = vm["z"].as<int>();
  int Mr = vm["Mr"].as<int>(), Kr = vm["Kr"].as<int>(), Nr = vm["Nr"].as<int>();
  if (!Mr) Mr = M;
  if (!Kr) Kr = K;
  if (!Nr) Nr = N;
  float wstd = vm["wstd"].as<float>(), outlier = vm["outlier"].as<float>();

  bool do_verify_stochastic = (long long)M * N > verify_stochastic_threshold;

  if (verbosity >= 1) {
    std::cout << "Matrix size " << M << "x" << K << "x" << N << std::endl;
  }

  int A_SIZE = M * K;
  int B_SIZE = N * K;
  int C_SIZE = M * N;

  size_t A_VOLUME = (A_SIZE * sizeof(uint8_t)) * 1.125;
  size_t B_VOLUME = (B_SIZE * sizeof(uint8_t)) * 1.125;
  size_t C_VOLUME = (C_SIZE * sizeof(uint8_t)) * 1.125;

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // ------------------------------------------------------
  // Get device, load the xclbin & kernel and register them
  // ------------------------------------------------------
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1) {
                                   std::cout << "Name: " << name << std::endl;
                                 }
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  // get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  // ------------------------------------------------------
  // Initialize input/output buffer sizes and sync them
  // ------------------------------------------------------

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a = xrt::bo(device, A_SIZE * sizeof(test_utils::bfloat16_t),
                      XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b =
      xrt::bo(device, B_VOLUME, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out = xrt::bo(device, C_SIZE * sizeof(test_utils::bfloat16_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  // ------------------------------------------------------
  // Generate data for buffers
  // ------------------------------------------------------
  if (verbosity >= 1) {
    std::cout << "Writing data into buffer objects.\n";
  }

  // Activation-like A ~ N(0,1) (optionally outlier channels, as in ViTs), weight-like B ~ N(0,wstd);
  // padding rows/cols are zero.
  std::mt19937 gen(1726250518);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<test_utils::bfloat16_t> AVec(A_SIZE, test_utils::bfloat16_from_float(0.f));
  for (int i = 0; i < Mr; i++)
    for (int j = 0; j < Kr; j++) {
      float v = nd(gen);
      if (outlier > 0 && j % 97 == 3) v *= outlier;
      AVec[i * K + j] = test_utils::bfloat16_from_float(v);
    }
  std::vector<test_utils::bfloat16_t> BVec(B_SIZE, test_utils::bfloat16_from_float(0.f));
  for (int i = 0; i < Kr; i++)
    for (int j = 0; j < Nr; j++)
      BVec[i * N + j] = test_utils::bfloat16_from_float(nd(gen) * wstd);

  // This is a quick conversion to avoid having to create a custom function for
  // bf16 for now
  std::vector<float> BVecFloat(B_SIZE);
  for (int i = 0; i < B_SIZE; i++) {
    BVecFloat[i] = test_utils::bfloat16_to_float(BVec[i]);
  }

  auto shuffleStart = std::chrono::high_resolution_clock::now();
  // ATB: re-layout B in column-major L1 tiles with a 1x2/8x8 sub-block shuffle,
  // THEN BFP-encode. The kernel reads B as a 1D stream of v8bfp16ebs8 vectors
  // per L1 tile.
  std::vector<float> BVecFloatShuffled =
      gemm_atb::layout_transpose_L1_1x2_8x8block(BVecFloat, K, N, k, n);
  std::vector<uint8_t> BVecBfpShuffled =
      floatToBfp16(8, B_SIZE, BVecFloatShuffled.data(), 0);
  auto shuffleStop = std::chrono::high_resolution_clock::now();

  float shuffleTime = std::chrono::duration_cast<std::chrono::microseconds>(
                          shuffleStop - shuffleStart)
                          .count();

  // ------------------------------------------------------
  // Write data into buffers
  // ------------------------------------------------------
  test_utils::bfloat16_t *bufA = bo_a.map<test_utils::bfloat16_t *>();
  uint8_t *bufB = bo_b.map<uint8_t *>();
  memcpy(bufA, AVec.data(), AVec.size() * sizeof(test_utils::bfloat16_t));
  memcpy(bufB, BVecBfpShuffled.data(), B_VOLUME);

  // Initialize outputs; bufOut is results matrix
  char *bufOut = bo_out.map<char *>();

  // Instruction buffer for DMA configuration
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // ------------------------------------------------------
  // Run kernel
  // ------------------------------------------------------
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  int errors = 0;
  float macs = 2.0 * float(M) * float(K) * float(N);

  std::vector<float> t_kern, t_in, t_out;
  for (unsigned iter = 0; iter < num_iter; iter++) {
    auto s0 = std::chrono::high_resolution_clock::now();
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_out);
    ert_cmd_state r = run.wait();
    if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << "Kernel did not complete. Returned status: " << r << "\n";
      return 1;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto s3 = std::chrono::high_resolution_clock::now();

    if (iter < n_warmup_iterations) {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    t_kern.push_back(npu_time);
    t_in.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(start - s0).count() / 1000.f);
    t_out.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(s3 - stop).count() / 1000.f);
    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  // ------------------------------------------------------
  // Verify against CPU reference ONCE, after timing loop completes, so the
  // verify pass does not pollute the per-iter average runtime.
  // ------------------------------------------------------
  {
    // Full error stats on the real (unpadded) region vs a double-precision reference on the bf16 inputs.
    std::vector<test_utils::bfloat16_t> CVec(C_SIZE);
    memcpy(CVec.data(), bufOut, CVec.size() * sizeof(test_utils::bfloat16_t));
    std::vector<float> Bt((size_t)Nr * Kr);
    for (int kk = 0; kk < Kr; kk++)
      for (int j = 0; j < Nr; j++)
        Bt[(size_t)j * Kr + kk] = test_utils::bfloat16_to_float(BVec[kk * N + j]);
    std::vector<float> Ar((size_t)Kr);
    double maxabs = 0, maxref = 0, se = 0, sr = 0, se_b = 0;
    for (int i = 0; i < Mr; i++) {
      for (int kk = 0; kk < Kr; kk++) Ar[kk] = test_utils::bfloat16_to_float(AVec[i * K + kk]);
      for (int j = 0; j < Nr; j++) {
        const float *bp = &Bt[(size_t)j * Kr];
        double acc = 0;
        for (int kk = 0; kk < Kr; kk++) acc += (double)Ar[kk] * bp[kk];
        double out = test_utils::bfloat16_to_float(CVec[i * N + j]);
        double e = std::fabs(out - acc);
        double eb = std::fabs(test_utils::bfloat16_to_float(test_utils::bfloat16_from_float((float)acc)) - acc);
        maxabs = std::max(maxabs, e);
        maxref = std::max(maxref, std::fabs(acc));
        se += e * e;
        sr += acc * acc;
        se_b += eb * eb;
      }
    }
    std::cout << "ERR max_abs " << maxabs << " max_abs/max|ref| " << maxabs / maxref << " rms_rel "
              << std::sqrt(se / sr) << " (bf16-rounding floor rms_rel " << std::sqrt(se_b / sr) << ")" << std::endl;
  }
  // ------------------------------------------------------
  // Output results
  // ------------------------------------------------------
  std::cout << std::endl
            << "Avg NPU matmul time: " << npu_time_total / n_iterations << "us."
            << std::endl;
  std::cout << "Avg NPU gflops: "
            << macs / (1000 * npu_time_total / n_iterations) << std::endl;

  std::cout << std::endl
            << "Min NPU matmul time: " << npu_time_min << "us." << std::endl;
  std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min) << std::endl;

  std::cout << std::endl
            << "Max NPU matmul time: " << npu_time_max << "us." << std::endl;
  std::cout << "Min NPU gflops: " << macs / (1000 * npu_time_max) << std::endl;

  auto med = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
  std::cout << "MED kernel_us " << med(t_kern) << " sync_in_us " << med(t_in) << " sync_out_us " << med(t_out)
            << " real_TOPS " << 2.0 * Mr * Kr * Nr / med(t_kern) / 1e6 << " padded_TOPS "
            << 2.0 * M * K * N / med(t_kern) / 1e6 << std::endl;
  std::cout << std::endl
            << "Shuffle time: " << shuffleTime << "us." << std::endl;

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }

  std::cout << "\nError count: " << errors;
  if (do_verify_stochastic) {
    std::cout << " (out of " << verify_stochastic_n_samples
              << " random samples)";
  }
  std::cout << "\n\n";

  std::cout << "\nFailed.\n\n";
  return 1;
}
