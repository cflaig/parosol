/*
 * PCGSolverMC.h
 *
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

#ifndef PCGSOLVERMC_H_
#define PCGSOLVERMC_H_

#include <iostream>
#include <cuda_runtime.h>
#include "VectorOperations.h"
#include "Solver.h"

#include "StiffnessMatrix.h"

//! This class implements an PCG-method.

/*! The preconditioned conjugate gradient method is a solver for symmetric
 *  positive definite linear system. The implementation is optimized in respect
 *  to low memory usage.
*/

class PCGSolverMC: public Solver
{
  public:

      /**
       * @brief Constructor of the PCG solver.
       * If you use it to apply the inverse of the matrix choose zeroStart to be true (example as Preeconditioner)
       *
       * @param M Stiffnessmatrix
	   * @param precon Preconditioner that has to implement the solver interface.
	   * @param tol Relative tolerance for solving.
	   * @param maxIter Maximal number of iteration of the iterative solving process.
	   * @param output If this parameter is set to true, the solver outputs the residual
	   * @param freq If output set it writes freq-th iteration the residual
	   *
	   */
    PCGSolverMC(StiffnessMatrix &M, Solver &precon,DispVector &dx,DispVector &dd, double tol=1e-10, int maxIter=500, bool output=true, int freq = 4);
    virtual ~PCGSolverMC();

    /**
     * @brief Solves the linear system \f[Ax=b\f] iteratively with the initial guess \f[x\f]
     * @param[in] b RHS
     * @param[in,out] x initial guess and solution
     */
    virtual int Solve(DispVector &b,DispVector &x,cudaStream_t streamB,cudaStream_t streamX);
    virtual const std::string  Label () const;
    void WaitOnStream(cudaStream_t waitOn,cudaStream_t stream);

  protected:
    StiffnessMatrix &_mat;
    Solver &_prec;
    long _ldofs;
    long _maxIter;
    double _tol;
    bool _verbose;
	int _freq;
	DispVector &x;
	DispVector &d;
	t_double *t_x;
	t_double *t_d;
	bool _first;
};



#endif /* PCGSOLVERMC_H_ */
