/* Selectable Chebyshev eigenvalue estimation.
 * Adaptive strategy and hashed start adapted from Cyril Flaig's ParOSol
 * commit c53b7b33d8df2baa18ed7903de8fd93161629001 (GPL-2.0-or-later).
 */
#ifndef PAROSOL_SPECTRAL_OPTIONS_H
#define PAROSOL_SPECTRAL_OPTIONS_H
#include <cstdint>
#include <cstdlib>
#include <string>
#include <memory>
#include <vector>

enum class SpectralMode { Original, Adaptive };
struct SpectralOptions {
  SpectralMode mode = SpectralMode::Adaptive;
  int adaptiveMaxIterations = 100;
  bool maxWasSpecified = false;
  std::shared_ptr<std::vector<std::string>> levelDiagnostics = std::make_shared<std::vector<std::string>>();
  int effectiveLimit() const {
    return mode == SpectralMode::Original ? 10 : adaptiveMaxIterations;
  }
};
inline const char *SpectralModeName(SpectralMode mode) {
  return mode == SpectralMode::Original ? "original" : "adaptive";
}
inline bool ParseSpectralMode(const std::string &s, SpectralMode &mode) {
  if (s == "original") mode = SpectralMode::Original;
  else if (s == "adaptive") mode = SpectralMode::Adaptive;
  else return false;
  return true;
}
inline bool IsSpectralModeOption(const std::string &token) {
  const std::string name = token.substr(0, token.find('='));
  return name == "--eig_estimator_mode" || name == "--eig-estimator-mode" ||
         name == "--eigen-estimator";
}
inline bool ParseSpectralMaximum(const std::string &s, int &limit) {
  if (s.empty()) return false;
  for (char c : s) if (c < '0' || c > '9') return false;
  if (s.size() > 4) return false;
  const long n = std::strtol(s.c_str(), nullptr, 10);
  if (n < 2 || n > 1000) return false;
  limit = static_cast<int>(n);
  return true;
}
inline double SpectralRandomValue(std::uint64_t key, unsigned component) {
  std::uint64_t v = key + 0x9e3779b97f4a7c15ULL * (component + 1);
  v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
  v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
  v ^= v >> 31;
  return 2.0 * double(v >> 11) * 0x1.0p-53 - 1.0;
}
#endif
