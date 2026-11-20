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


#ifndef MLLEVELCG_H 
#define MLLEVELCG_H 

#include "Solver.h"
#include "StiffnessMatrix.h"

#include "Jacobi.h"
#include "PCGSolver.h"

/*! This class wraps a PCG solver for coarse level solver */
class MlLevelCG: public Solver
{
  public:
    MlLevelCG(StiffnessMatrix &M,GPUMemoryManager *gpuMM):prec(M,gpuMM),solver(M,prec,1e-7,20,false)
    {
    	solver.SetGPUMemoryManager(gpuMM);
    }
    virtual ~MlLevelCG()
    {
    }

    virtual int Solve(DispVector &b,DispVector &x,cudaStream_t streamB,cudaStream_t streamX) {
      return solver.Solve(b,x,streamB,streamX);
    }

    const std::string Label() const
    {
      return "Coarsegrid PCG solver\n";
    }
  private:
    Jacobi prec;
    PCGSolver solver;

};
#endif /* MLLEVELSOLVERCG_H  */
