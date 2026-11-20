#ifndef PAROSOL_SOLVE_DIAGNOSTICS_H
#define PAROSOL_SOLVE_DIAGNOSTICS_H
#include "SpectralOptions.h"
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

inline std::string DiagnosticNumber(double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream out; out << std::setprecision(17) << value; return out.str();
}
struct SolveDiagnostics {
  SpectralOptions spectral;
  const char *backend = "cpu";
  const char *solveScope = "pcg_excludes_final_residual_evaluation";
  int iterations = 0;
  double cold = 0, initial = 0, final = 0, tolerance = 0;
  double preconditionerSeconds = 0, solveSeconds = 0, cacheLoadSeconds = 0, cacheSaveSeconds = 0;
  bool warmRequested = false, warmEnabled = false, warmHit = false, warmSaved = false;
  bool additionalFiniteCheck = true;
  double relativeResidual() const {
    return cold > 0 ? final / cold : final == 0 ? 0 : std::numeric_limits<double>::infinity();
  }
  bool converged() const {
    return additionalFiniteCheck && std::isfinite(cold) && cold >= 0 &&
      std::isfinite(final) && final >= 0 && std::isfinite(tolerance) && tolerance > 0 &&
      (cold == 0 ? final == 0 : final <= tolerance * cold);
  }
  const char *reason() const {
    if (!additionalFiniteCheck || !std::isfinite(cold) || !std::isfinite(final)) return "nonfinite_residual_or_solution";
    return converged() ? "converged" : "residual_above_tolerance";
  }
  std::string json() const {
    const char *warmReason = !warmRequested ? "disabled" : !warmEnabled ? "unsupported_multi_rank" :
      warmHit ? "compatible_previous_solution" : "no_compatible_previous_solution";
    std::ostringstream out;
    out << std::boolalpha << "{\"schema_version\":2,\"backend\":\"" << backend
        << "\",\"eig_estimator_mode\":\"" << SpectralModeName(spectral.mode)
        << "\",\"estimator_cap\":" << spectral.effectiveLimit() << ",\"spectral_levels\":[";
    for (std::size_t i=0;i<spectral.levelDiagnostics->size();++i) {
      if (i) out << ',';
      out << (*spectral.levelDiagnostics)[i];
    }
    out << "],\"pcg_iterations\":" << iterations
        << ",\"cold_residual\":" << DiagnosticNumber(cold)
        << ",\"initial_residual\":" << DiagnosticNumber(initial)
        << ",\"final_residual\":" << DiagnosticNumber(final)
        << ",\"relative_residual\":" << DiagnosticNumber(relativeResidual())
        << ",\"normalization\":\"cold_initial_residual_norm\",\"tolerance\":" << DiagnosticNumber(tolerance)
        << ",\"converged\":" << converged() << ",\"termination_reason\":\"" << reason()
        << "\",\"status\":" << (converged() ? 0 : 2)
        << ",\"preconditioner_seconds\":" << DiagnosticNumber(preconditionerSeconds)
        << ",\"solve_seconds\":" << DiagnosticNumber(solveSeconds)
        << ",\"solve_timer_scope\":\"" << solveScope
        << "\",\"warm_requested\":" << warmRequested << ",\"warm_enabled\":" << warmEnabled
        << ",\"warm_hit\":" << warmHit << ",\"warm_saved\":" << warmSaved
        << ",\"warm_reason\":\"" << warmReason
        << "\",\"cache_load_seconds\":" << DiagnosticNumber(cacheLoadSeconds)
        << ",\"cache_save_seconds\":" << DiagnosticNumber(cacheSaveSeconds) << '}';
    return out.str();
  }
};
#endif
