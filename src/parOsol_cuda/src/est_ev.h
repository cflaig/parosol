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


#ifndef EST_EV_H
#define EST_EV_H

#include <cuda_runtime.h>
#include <iostream>
#include "Solver.h"
#include "StiffnessMatrix.h"
#include "VectorOperations.h"

struct eig {
  double large = 0, small = 0;
  int iterations = 0;
  double relativeResidual = 0, relativeChange = 0;
  bool converged = false, invariantSubspace = false;
};

/** computes the eigenvalues of the preconditions system
 * 
 * @param _mat the system matrix
 * @param _prec preconditioner
 * @param maxint number of iteration to compute the eigenvalues
 * @param m1 temporary Vector, that is needed for the computation.
 * @param m2 temporary Vector, that is needed for the computation.
 * @param gpuMM the gpu memory manager object
 */

eig est_ev(StiffnessMatrix &_mat, Solver &_prec, int maxint,DispVector &m1,DispVector &m2,GPUMemoryManager *gpuMM);
eig EstimateChebyshevSpectrum(StiffnessMatrix &mat, int maxit,
                             DispVector &work1, DispVector &work2,
                             GPUMemoryManager *gpuMM);
#endif /* EST_EV_H */
