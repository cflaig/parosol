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

#ifndef PROBLEM_H
#define PROBLEM_H

#include <cuda_runtime.h>
#include "BoundaryCondition.h"
#include "Solver.h"
#include "Config.h"
#include "OtherKernels.h"
#include "VectorOperations.h"
#include "GPUMemoryManager.h"



//! This class holds the linear system 

/*! Problem stores the Matrix, RHS and LHS. It also computes the the
 *  LHS out of the boundary conditions
 */

template <class Grid>
class Problem
{
	public:
		//!Problem constructor

		Problem(GenericMatrix<Grid> &mat,GPUMemoryManager *gpuMM):
			 _mat(mat), _bcond(mat.GetBC())
		{

			_gpuMemoryManager = gpuMM;
			//allocate the LHS and RHS. Value are set through the BC
			if (_mat.GetSyncData().switching) {
				x = new DispVectorS(_mat.GetSyncData(),_gpuMemoryManager);
				b = new DispVectorS(_mat.GetSyncData(),_gpuMemoryManager);
			} else {
				x = new DispVectorN(_mat.GetSyncData(),_gpuMemoryManager);
				b = new DispVectorN(_mat.GetSyncData(),_gpuMemoryManager);
			}
			tmpB = new t_double[b->_syncData.gpuVecSize];
		}

		//! Problem destructor
		~Problem() {
			delete x;
			delete b;
			delete tmpB;
		}

		void TestApply() {
			x->Memset(1.0);
			for (int i = 0;i < 100;i++) {
				_mat.Apply(*x,*b);
				_mat.Apply(*b,*x);
			}
		}

		void TestAdd() {
			for (int i = 0;i < 5000;i++) {
				AddVector(*x,*b);
			}
		}

		//! Impose boundary condition
		int Impose();
		
		/** Set the solver
		 * @param s Implemented Solver
		 */
		void SetSolver(Solver &s) {
			solver = &s;
		}

		//! Start the solver
		int Solve() {
			// The solver may change b
			b->CopyFromExtCpuVector(tmpB);

			cudaStream_t streamB;
			cudaStream_t streamX;
			CCS(cudaStreamCreate(&streamB),"Solve 1");
			CCS(cudaStreamCreate(&streamX),"Solve 2");

			int res = solver->Solve(*b,*x,streamB,streamX);

			CCS(cudaDeviceSynchronize(),"Solve 3");
			CCS(cudaStreamDestroy(streamB),"Solve 4");
			CCS(cudaStreamDestroy(streamX),"Solve 5");

			return res;
		}

		/** Gets the solution
		 * @return a refence to the solution vector
		 */
        t_double * GetSol() {
        	t_double *tmp = new t_double[x->_syncData.gpuVecSize];
        	x->CopyToExtCpuVector(tmp);
        	return tmp;
		}

        DispVector& GetSolDev() {
        	return *x;
        }
		
		/** Gets the residual vector
		 * @return a refence to a the residual vector
		 */
        t_double * GetRes() {

        	_mat.Apply(*x,*b);
        	t_double *tmp = new t_double[b->_syncData.gpuVecSize];
        	b->CopyToExtCpuVector(tmp);
        	for (uint i = 0;i < b->_syncData.gpuVecSize;i++) {
        		tmp[i] = tmpB[i] - tmp[i];
        	}

        	return tmp;
		}
		
		/** computes the true residuum
		 * @return the residuum
		 */
		double Res() {

			_mat.Apply(*x,*b);

			t_double *tmp = new t_double[b->_syncData.gpuVecSize];
			b->CopyToExtCpuVector(tmp);

			for (uint i = 0;i < b->_syncData.gpuVecSize;i++) {
				tmp[i] = tmpB[i] - tmp[i];
			}

			b->CopyFromExtCpuVector(tmp);

			double res = (double) _mat.dot(*b);
			delete[] tmp;
			return sqrt(res);
		}

		
				
		/** @name Some Debug function */
		//@{
  		/** Compares a vector with the solution
		 * @param x vector that is compared to the solution of the problem
		 * @return the norm of the difference
		 */
		double CompareSol(t_double *tx) {
			t_double *tmp = new t_double[x->_syncData.gpuVecSize];
			x->CopyToExtCpuVector(tmp);
			tmp = tx - tmp;
			b->CopyFromExtCpuVector(tmp);
			delete tmp;
			return sqrt(_mat.dot(*b));
		}
		//! Print solutioin vector
		void PrintSol() {
			std::cout << "\nSolution\n";
			t_double *tmp = new t_double[x->_syncData.gpuVecSize];
			x->CopyToExtCpuVector(tmp);
			_mat.GetGrid().PrintVector(tmp);
			delete tmp;
		}

		//! Print the RHS
		void PrintRHS() {
			std::cout << "\n RHS \n";
			t_double *tmp = new t_double[b->_syncData.gpuVecSize];
			b->CopyToExtCpuVector(tmp);
			_mat.GetGrid().PrintVector(tmp);
			delete tmp;
		} //@}

	private:
		//! Helper function which set BC on the RHS
		int SetBoundaryConditions(DispVector &rhs);
		GenericMatrix<Grid> &_mat;
		BoundaryCondition &_bcond;
		DispVector *x;
		DispVector *b;
		Solver *solver;
		GPUMemoryManager *_gpuMemoryManager;
		t_double *tmpB;
};

template <class Grid>
int Problem<Grid>::Impose() {

	//renaming
	x->Memset(0.0);
	b->Memset(0.0);
	//Store Loads in B
	//At the moment none load supported

	//apply fixed nodes on Matrix
	//write displacement in x
	//corr = A*fixed_disp -> load which is cause by the fixed nodes 
	SetBoundaries3K(_bcond,*x);

	_mat.Apply_NoResetBoundaries(*x,*b);

	ScaleVector(-1,*b);

    //add the loaded nodes
	AddBoundariesK(_bcond,*b);

	SetBoundaryConditions(*b);

	//x.setRandom(_ldofs);
	//x.setConstant(_ldofs, 0.02);
	x->Memset(0.0);
	SetBoundaries3K(_bcond,*x);

	b->CopyToExtCpuVector(tmpB);

	return 0;
}

template <class Grid>
int Problem<Grid>::SetBoundaryConditions(DispVector &rhs) {
	SetBoundaries3K(_bcond,rhs);
	// write the disp in the vector 
	return 0;
}

#endif /* PROBLEM_H */

