// Dispatch timing from C++ (no Python in the loop): same xclbin / instruction
// buffer / buffer sizes as run.py, zero-filled data, median submit+wait.
//   host XCLBIN INSTR_BIN A_BYTES W_BYTES C_BYTES ITERS [WARMUP]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

int main(int argc, char **argv) {
  if (argc < 7) {
    std::fprintf(stderr, "usage: host XCLBIN INSTR A_BYTES W_BYTES C_BYTES ITERS [WARMUP]\n");
    return 2;
  }
  const char *xclbin_path = argv[1], *instr_path = argv[2];
  size_t a_bytes = std::strtoull(argv[3], 0, 10), w_bytes = std::strtoull(argv[4], 0, 10),
         c_bytes = std::strtoull(argv[5], 0, 10);
  int iters = std::atoi(argv[6]), warmup = argc > 7 ? std::atoi(argv[7]) : 3;

  std::ifstream f(instr_path, std::ios::binary);
  std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  size_t n_instr = raw.size() / 4;

  xrt::device device(0);
  xrt::xclbin xclbin{std::string(xclbin_path)};
  device.register_xclbin(xclbin);
  xrt::hw_context ctx(device, xclbin.get_uuid());
  auto kname = xclbin.get_kernels()[0].get_name();
  xrt::kernel kernel(ctx, kname);

  xrt::bo bo_instr(device, raw.size(), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  xrt::bo bo_a(device, a_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  xrt::bo bo_w(device, w_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  xrt::bo bo_c(device, c_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  bo_instr.write(raw.data());
  std::vector<char> zeros(std::max({a_bytes, w_bytes, c_bytes}), 0);
  bo_a.write(zeros.data(), a_bytes, 0);
  bo_w.write(zeros.data(), w_bytes, 0);
  bo_c.write(zeros.data(), c_bytes, 0);
  for (auto *b : {&bo_instr, &bo_a, &bo_w, &bo_c}) b->sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::vector<double> us;
  for (int it = 0; it < warmup + iters; it++) {
    auto t0 = std::chrono::steady_clock::now();
    auto run = kernel(3, bo_instr, n_instr, bo_a, bo_w, bo_c);
    auto st = run.wait();
    auto t1 = std::chrono::steady_clock::now();
    if (st != ERT_CMD_STATE_COMPLETED) {
      std::printf("RUN FAILED state %d\n", (int)st);
      return 1;
    }
    if (it >= warmup) us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  std::sort(us.begin(), us.end());
  std::printf("CPP kernel_us median %.0f min %.0f (n=%d, instr words %zu)\n", us[us.size() / 2], us[0],
              iters, n_instr);
  return 0;
}
