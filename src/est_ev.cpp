/*
 * ParOSol: a parallel FE solver for trabecular bone modeling
 * Copyright (C) 2011, Cyril Flaig
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */




#include "est_ev.h"
#include "Jacobi.h"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>

namespace {
constexpr int checkInterval = 5;
constexpr double relativeTolerance = 0.01;

void requirePositive(double value, const char *message) {
  if (!(value > 0) || !std::isfinite(value)) throw std::runtime_error(message);
}
}

eig est_ev(StiffnessMatrix &mat, Solver &prec, int maxit,
           Eigen::VectorXd &r, Eigen::VectorXd &d) {
  if (maxit < 1) throw std::invalid_argument("Eigenvalue iteration limit must be positive");
  Eigen::VectorXd diagonal(maxit), offDiagonal(maxit);
  mat.SetVectorRandom(r);
  prec.Solve(r, d);
  Eigen::VectorXd work(mat.GetNrDofs());
  double delta = mat.dot(r, d);
  requirePositive(delta, "Eigenvalue estimate: zero or nonpositive starting norm");
  const double initialDelta = delta;
  double alpha = 0, beta = 0, previousTheta = 0;
  int consecutive = 0;
  eig result;

  for (int k = 0; k < maxit; ++k) {
    mat.Apply(d, work);
    const double curvature = mat.dot(d, work);
    requirePositive(curvature, "Eigenvalue estimate: nonpositive matrix curvature");
    const double previousAlpha = alpha;
    alpha = delta / curvature;
    requirePositive(alpha, "Eigenvalue estimate: invalid CG step");
    diagonal[k] = 1 / alpha;
    if (k > 0) {
      diagonal[k] += beta / previousAlpha;
      offDiagonal[k-1] = std::sqrt(beta) / previousAlpha;
    }

    r -= alpha * work;
    prec.Solve(r, work);
    const double nextDelta = mat.dot(r, work);
    if (!(nextDelta >= 0) || !std::isfinite(nextDelta))
      throw std::runtime_error("Eigenvalue estimate: invalid preconditioned residual");
    beta = nextDelta / delta;
    if (!std::isfinite(beta)) throw std::runtime_error("Eigenvalue estimate: invalid CG recurrence");
    d = work + beta * d;
    delta = nextDelta;
    const int steps = k + 1;
    const double eps = std::numeric_limits<double>::epsilon();
    const bool terminated = nextDelta / initialDelta <= eps * eps;
    if (steps % checkInterval != 0 && steps != maxit && !terminated) continue;

    // Only the small tridiagonal eigenvectors are needed. No large Krylov
    // basis and no additional applications of the FE operator are stored/used.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;
    solver.computeFromTridiagonal(diagonal.head(steps), offDiagonal.head(steps-1),
                                  Eigen::ComputeEigenvectors);
    if (solver.info() != Eigen::Success)
      throw std::runtime_error("Eigenvalue estimate: tridiagonal eigensolve failed");
    result.small = solver.eigenvalues()[0];
    result.large = solver.eigenvalues()[steps-1];
    requirePositive(result.large, "Eigenvalue estimate: invalid largest Ritz value");
    // The last Lanczos coupling is sqrt(beta_CG)/alpha_CG.
    const double residual = std::abs(std::sqrt(beta) / alpha *
                                    solver.eigenvectors()(steps-1,steps-1));
    result.relativeResidual = residual / result.large;
    result.relativeChange = previousTheta > 0 ?
        std::abs(result.large - previousTheta) / result.large :
        1.0;
    result.iterations = steps;
    result.invariantSubspace = terminated;
    if (steps % checkInterval == 0) {
      const bool passed = previousTheta > 0 &&
          result.relativeChange < relativeTolerance &&
          result.relativeResidual < relativeTolerance;
      consecutive = passed ? consecutive + 1 : 0;
      previousTheta = result.large;
    }
    result.converged = consecutive >= 2 ||
        (terminated && result.relativeResidual < relativeTolerance);
    if (result.converged || terminated) return result;
  }
  return result; // The caller must handle an unconverged iteration limit.
}

eig EstimateChebyshevSpectrum(StiffnessMatrix &mat, int maxit,
                             Eigen::VectorXd &work1, Eigen::VectorXd &work2) {
  Jacobi prec(mat);
  const eig e = est_ev(mat, prec, maxit, work1, work2);
  if (mat.GetPID() == 0)
    std::cout << "Spectral estimate: iterations=" << e.iterations
              << " limit=" << maxit << " converged=" << e.converged
              << " relative_ritz_residual=" << e.relativeResidual
              << " relative_change=" << e.relativeChange
              << " invariant_subspace=" << e.invariantSubspace << std::endl;
  if (!e.converged)
    throw std::runtime_error("Chebyshev eigenvalue estimate did not converge; increase --eigen-max-iterations");
  return e;
}
