/*
 * GParOSol: a parallel FE solver for trabecular bone modeling
 * Copyright (C) 2011, Cyril Flaig
 * Copyright (C) 2013, Daniel Kellenberger
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
#include "float.h"
#include "tri_eig.h"
#include "OtherKernels.h"
#include "DispVectorN.h"

/// Compute the EV of a tridiagonal symmetric matrix
int QL_Tridiagonal_Symmetric_Matrix(t_double *diagonal,t_double *p_off,
                                   int n, int max_iteration_count);

eig est_ev(StiffnessMatrix &_mat, Solver &_prec, int maxit,DispVector &r,DispVector &d,GPUMemoryManager *gpuMM) {

	double alpha = 0, beta =0; // in first iteration beta_old is not used
	double alpha_old, res0, delta_new, delta_old;
	double _tol = 1e-7;
	t_double *dia_T = new t_double[maxit+1];
	t_double *off_T = new t_double [maxit+1];
	r.Memset(1.0);
	int i = 0;

	_prec.Solve(r,d,0,0);

	DispVector *s;
	if (_mat.GetSyncData().switching) {
		s = new	DispVectorS(_mat.GetSyncData(),gpuMM);
	} else {
		s = new DispVectorN(_mat.GetSyncData(),gpuMM);
	}
	s->Copy(r);

	res0 = delta_new = _mat.dot(r,d);

	res0 = sqrt(res0);
	while ( i < maxit && sqrt(delta_new)/res0 > _tol) {
		_mat.Apply(d,*s);

		alpha_old = alpha;
		alpha = delta_new/_mat.dot(d,*s);
		//compute tridiagonal matrix
		dia_T[i] = 1/alpha;
		if (i > 0) {
			dia_T[i] += beta/alpha_old;
			off_T[i-1] = sqrt(beta)/alpha_old;
		}
		Axpy((-1)*alpha,*s,r);

		_prec.Solve(r,*s,0,0);
		delta_old = delta_new;
		delta_new = _mat.dot(r,*s);

		beta = delta_new/delta_old;
		Axpy2(beta,d,*s);

		i++;
	}
	double l_min, l_max;
	if (i == 0) {
			// Degenerate/near-empty level: the very first residual check
			// already satisfied tolerance (or produced 0/0 = NaN, which
			// also fails the "> _tol" check), so no Lanczos iterations
			// ran. dia_T/off_T hold no data here -- calling
			// QL_Tridiagonal_Symmetric_Matrix with n=0 writes to
			// p_off[-1] (heap corruption, surfaces later as "free():
			// invalid pointer"), and dia_T[0] would be read
			// uninitialised. Nothing to smooth at this level; fall back
			// to a trivial eigenvalue bound.
			l_min = l_max = 1.0;
	} else {
			QL_Tridiagonal_Symmetric_Matrix(dia_T, off_T,i,30);
			l_min = l_max = dia_T[0];
			for(int ii = 1; ii < i; ii++) {
			if (dia_T[ii] < l_min)
				l_min = dia_T[ii];
			if (dia_T[ii] > l_max)
				l_max = dia_T[ii];
			}
	}
	eig ret;
	ret.iterations=i;
	ret.large=l_max;
	ret.small=l_min;
	delete[] dia_T;
	delete[] off_T;
	delete s;
	return ret;
}


// Adapted from Cyril Flaig c53b7b33, using CUDA/host DispVector operations.
#include "Jacobi.h"
#include "DispVectorS.h"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {
constexpr int checkInterval = 5;
constexpr double relativeTolerance = 0.01;
void requirePositive(double value, const char *message) {
  if (!(value > 0) || !std::isfinite(value)) throw std::runtime_error(message);
}
}

static eig est_ev_adaptive(StiffnessMatrix &mat, Solver &prec, int maxit,
                           DispVector &r, DispVector &d, GPUMemoryManager *gpuMM) {
  if (maxit < 1) throw std::invalid_argument("Eigenvalue iteration limit must be positive");
  Eigen::VectorXd diagonal(maxit), offDiagonal(maxit);
  mat.SetVectorRandom(r);
  prec.Solve(r, d, 0, 0);
  std::unique_ptr<DispVector> work;
  if (mat.GetSyncData().switching)
    work.reset(new DispVectorS(mat.GetSyncData(), gpuMM));
  else
    work.reset(new DispVectorN(mat.GetSyncData(), gpuMM));
  double delta = mat.dot(r, d);
  requirePositive(delta, "Eigenvalue estimate: zero or nonpositive starting norm");
  const double initialDelta = delta;
  double alpha = 0, beta = 0, previousTheta = 0;
  int consecutive = 0;
  eig result;
  for (int k = 0; k < maxit; ++k) {
    mat.Apply(d, *work);
    const double curvature = mat.dot(d, *work);
    requirePositive(curvature, "Eigenvalue estimate: nonpositive matrix curvature");
    const double previousAlpha = alpha;
    alpha = delta / curvature;
    requirePositive(alpha, "Eigenvalue estimate: invalid CG step");
    diagonal[k] = 1 / alpha;
    if (k > 0) {
      diagonal[k] += beta / previousAlpha;
      offDiagonal[k-1] = std::sqrt(beta) / previousAlpha;
    }
    Axpy(-alpha, *work, r);
    prec.Solve(r, *work, 0, 0);
    const double nextDelta = mat.dot(r, *work);
    if (!(nextDelta >= 0) || !std::isfinite(nextDelta))
      throw std::runtime_error("Eigenvalue estimate: invalid preconditioned residual");
    beta = nextDelta / delta;
    if (!std::isfinite(beta)) throw std::runtime_error("Eigenvalue estimate: invalid CG recurrence");
    Axpy2(beta, d, *work);
    delta = nextDelta;
    const int steps = k + 1;
    const double eps = std::numeric_limits<double>::epsilon();
    const bool terminated = nextDelta / initialDelta <= eps * eps;
    if (steps % checkInterval != 0 && steps != maxit && !terminated) continue;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;
    solver.computeFromTridiagonal(diagonal.head(steps), offDiagonal.head(steps-1),
                                  Eigen::ComputeEigenvectors);
    if (solver.info() != Eigen::Success)
      throw std::runtime_error("Eigenvalue estimate: tridiagonal eigensolve failed");
    result.small = solver.eigenvalues()[0];
    result.large = solver.eigenvalues()[steps-1];
    requirePositive(result.large, "Eigenvalue estimate: invalid largest Ritz value");
    const double residual = std::abs(std::sqrt(beta) / alpha *
                                    solver.eigenvectors()(steps-1,steps-1));
    result.relativeResidual = residual / result.large;
    result.relativeChange = previousTheta > 0 ?
        std::abs(result.large - previousTheta) / result.large : 1.0;
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
  return result;
}

eig EstimateChebyshevSpectrum(StiffnessMatrix &mat, int maxit,
                             DispVector &work1, DispVector &work2,
                             GPUMemoryManager *gpuMM) {
  Jacobi prec(mat, gpuMM);
  const eig e = est_ev_adaptive(mat, prec, maxit, work1, work2, gpuMM);
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
