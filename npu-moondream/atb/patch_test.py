"""Patch the ATB mixed harness: real (unpadded) dims, Gaussian data, full error stats, sync timings."""

s = open("test.cpp").read()
BD = "/work/mlir-aie/programming_examples/ml/block_datatypes"
s = s.replace('#include "../helper.h"', f'#include "{BD}/helper.h"')
s = s.replace(
    '#include "gemm_atb_layout.h"',
    f'#include "{BD}/gemm_asymmetric_tile_buffering/gemm_atb_layout.h"\n#include <random>\n#include <algorithm>',
)
s = s.replace(
    '"tcolumns,z", "Tile size n", cxxopts::value<int>()->default_value("64"));',
    '''"tcolumns,z", "Tile size n", cxxopts::value<int>()->default_value("64"))(
      "Mr", "real M (rest is zero padding)", cxxopts::value<int>()->default_value("0"))(
      "Kr", "real K", cxxopts::value<int>()->default_value("0"))(
      "Nr", "real N", cxxopts::value<int>()->default_value("0"))(
      "wstd", "weight std", cxxopts::value<float>()->default_value("0.03"))(
      "outlier", "activation outlier channel scale (0 = none)", cxxopts::value<float>()->default_value("0"));''',
)
s = s.replace(
    '  int n = vm["z"].as<int>();\n',
    '''  int n = vm["z"].as<int>();
  int Mr = vm["Mr"].as<int>(), Kr = vm["Kr"].as<int>(), Nr = vm["Nr"].as<int>();
  if (!Mr) Mr = M;
  if (!Kr) Kr = K;
  if (!Nr) Nr = N;
  float wstd = vm["wstd"].as<float>(), outlier = vm["outlier"].as<float>();
''',
)
a = s.index("  std::vector<test_utils::bfloat16_t> AVec(A_SIZE);")
b = s.index("  // This is a quick conversion")
s = (
    s[:a]
    + '''  // Activation-like A ~ N(0,1) (optionally outlier channels, as in ViTs), weight-like B ~ N(0,wstd);
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

'''
    + s[b:]
)
s = s.replace(
    """  for (unsigned iter = 0; iter < num_iter; iter++) {
    auto start = std::chrono::high_resolution_clock::now();""",
    """  std::vector<float> t_kern, t_in, t_out;
  for (unsigned iter = 0; iter < num_iter; iter++) {
    auto s0 = std::chrono::high_resolution_clock::now();
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto start = std::chrono::high_resolution_clock::now();""",
)
s = s.replace(
    """    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < n_warmup_iterations) {""",
    """    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto s3 = std::chrono::high_resolution_clock::now();

    if (iter < n_warmup_iterations) {""",
)
s = s.replace(
    "    npu_time_total += npu_time;",
    """    t_kern.push_back(npu_time);
    t_in.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(start - s0).count() / 1000.f);
    t_out.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(s3 - stop).count() / 1000.f);
    npu_time_total += npu_time;""",
)
vs = s.index("  if (do_verify) {")
ve = s.index("  // ------------------------------------------------------\n  // Output results")
s = (
    s[:vs]
    + """  {
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
"""
    + s[ve:]
)
s = s.replace(
    """  std::cout << std::endl
            << "Shuffle time: \"""",
    """  auto med = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
  std::cout << "MED kernel_us " << med(t_kern) << " sync_in_us " << med(t_in) << " sync_out_us " << med(t_out)
            << " real_TOPS " << 2.0 * Mr * Kr * Nr / med(t_kern) / 1e6 << " padded_TOPS "
            << 2.0 * M * K * N / med(t_kern) / 1e6 << std::endl;
  std::cout << std::endl
            << "Shuffle time: \"""",
)
assert s.count("MED kernel_us") == 1 and s.count("ERR max_abs") == 1 and s.count("t_kern.push_back") == 1
assert "s3 - stop" in s and "Mr\", \"real M" in s and "outlier > 0" in s
open("test.cpp", "w").write(s)
print("patched")
