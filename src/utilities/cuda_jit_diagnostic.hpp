#pragma once

// Reports only the actionable case: the running GPU has no compatible native
// cubin in this build, so CUDA must use the embedded PTX fallback. The driver
// may compile that PTX on a cache miss or load a prior JIT result from its
// on-disk cache. The CUDA Runtime does not expose a generic "JIT happened"
// callback for statically registered modules, so this is deliberately named
// PTX_FALLBACK rather than JIT.

#ifdef USE_CUDA

#include <cuda_runtime_api.h>

#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

#ifndef IFB_CUDA_ARCHITECTURES_CSV
#define IFB_CUDA_ARCHITECTURES_CSV ""
#endif

namespace ifb::cuda_diagnostics {

inline bool has_compatible_native_cubin(const cudaDeviceProp &device) {
  std::istringstream architectures(IFB_CUDA_ARCHITECTURES_CSV);
  std::string architecture;
  while (std::getline(architectures, architecture, ',')) {
    try {
      const int sm = std::stoi(architecture);
      const int major = sm / 10;
      const int minor = sm % 10;
      // A cubin is compatible only within the same major capability, and a
      // cubin built for a lower minor can run on a higher minor of that major.
      if (major == device.major && minor <= device.minor)
        return true;
    } catch (...) {
      // Ignore malformed build metadata; this diagnostic must never affect a
      // simulation.
    }
  }
  return false;
}

inline void log_ptx_jit_requirement_once(std::ostream &out = std::cerr) {
  static std::once_flag once;
  std::call_once(once, [&out] {
    int device_index = 0;
    cudaDeviceProp device{};
    if (cudaGetDevice(&device_index) != cudaSuccess ||
        cudaGetDeviceProperties(&device, device_index) != cudaSuccess ||
        has_compatible_native_cubin(device)) {
      return;
    }
    out << "[CUDA_PTX_FALLBACK] required before first CUDA module use: device sm_"
        << device.major << device.minor
        << " has no compatible native cubin in this binary (sm_"
        << IFB_CUDA_ARCHITECTURES_CSV
        << "); CUDA will use the embedded PTX fallback (JIT compile on a "
           "cache miss, otherwise load the driver cache).\n";
  });
}

} // namespace ifb::cuda_diagnostics

#endif // USE_CUDA
