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

#ifndef SOLVER_H
#define SOLVER_H

#include <string>
#include <math.h>
#include "Config.h"
#include "VectorKernels.h"
#include "GPUMemoryManager.h"


//! An interface for solvers


class Solver 
{
public:

  virtual ~Solver() {};

  /** Solves \f[Ax=b\f]
   * @param [in] b rhs vector
   * @param [out] x solution vector
   */
  virtual int   Solve(DispVector &b,DispVector &x,cudaStream_t streamB,cudaStream_t streamX) = 0;

  /** Returns the name of the solver
   * 
   * @return return the name in an string
   */
  virtual const std::string  Label () const = 0;

  void SetGPUMemoryManager(GPUMemoryManager *gpuMM) {
	  _gpuMemoryManager = gpuMM;
  }

protected:
  GPUMemoryManager *_gpuMemoryManager;

};

#endif /* SOLVER_H */

