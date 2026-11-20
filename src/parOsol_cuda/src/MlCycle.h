/*
 * ParOSol: a parallel FE solver for trabecular bone modeling
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

#ifndef MLCYCLE_H
#define MLCYCLE_H

#include "Solver.h"
#include "JacobiSmoother.h"
#include "Chebyshev.h"
#include "SpectralOptions.h"
#include "OctreeGrid.h"
#include "MlOctreeGrid.h"
#include "MlLevelCG.h"
#include "MlLevelCheb.h"
#include "OtherKernels.h"
#include "VectorKernels.h"
#include "VectorOperations.h"

//! A class to perform an MG-Cycle 

/*! This is the MG-Algorithm. Attention: The matrix must be prepared (Must have
 * included the boundaries condition)

*/

//#define DEBOUT

template <typename T>
class MlCycle : public Solver
{
	public:
       /** 
       * @brief Constructor of the MG-Algorithm.
       * The matrix must be prepared. This means it must have included the boundaries.
       * 
       * @param finegrid The grid of the finest level.
	   * @param mat The matrix of the finest level.
	   * @param degree Degree of the smoother
	   * @param level Number of max level.
	   * @param stype Smoother type (0: Chebyshev, 1: Jacobismoother)
	   * @param ratio Ration of the Eigenvlaue. See Chebyshev smoother
       * @param w weight parameter of the Jacobismoother
	   */
		MlCycle(OctreeGrid<T> &finegrid, GenericMatrix<OctreeGrid<T> > &mat,Timer *timing,GPUGridBuilder<T> *gpuGridBuilder,GPUMemoryManager *gpuMM,bool switching = false,int degree =8, int level = -1, int stype =0, double ratio=16, double w=0.61, SpectralOptions spectral = SpectralOptions{});

		~MlCycle()
		{
			delete _coarsegrid;
			delete _coarsemat;
			delete _solver;
			delete presmoother;
			delete postsmoother;
			FreeRessources();
		}

		int Solve(DispVector &b,DispVector &x,cudaStream_t streamB,cudaStream_t streamX);

		const std::string Label() const
		{
			return "Multilevel solver";
		}

		MlOctreeGrid<T>& GetCoarseGrid() {
			return *_coarsegrid;
		}

		GenericMatrix<OctreeGrid<T> >& GetCoarseMat() {
			return *_coarsemat;
		}

		void PrintTimings() {
			PrintTimings(1);
			PrintTimings(2);
		}

		DispVector& GetEFine() {
			return *eFine;
		}

		DispVector& GetRFine() {
			return *rFine;
		}

		void FreeRessources() {
			if (!_ressourcesFreed) {
				delete rFine;
				delete eFine;
				delete rCoarse;
				delete eCoarse;
				_ressourcesFreed = true;
			}
		}

	private:

		void PrintTimings(int id) {
			t_timing elaSum;
			elaSum.min = 0;
			elaSum.avg = 0;
			elaSum.max = 0;
			int countSum = 0;
			if (id == 1) {
				PCOUT(_finemat.GetPID(),"Restrict:" << std::endl)
			} else {
				PCOUT(_finemat.GetPID(),"Prolongate:" << std::endl)
			}

			PCOUT(_finemat.GetPID(),"level; time avg; time min; time max; calls" << std::endl)
			for (int i = _finemat._lMax - 1;i >= 0;i--) {
				t_timing ela = _timing->ElapsedTime(id,i);
				int count = _timing->Counter(id,i);
				PCOUT(_finemat.GetPID(),i << "; " << ela.avg << "; " << ela.min << "; " << ela.max << "; " << count << std::endl)
				elaSum.min += ela.min;
				elaSum.avg += ela.avg;
				elaSum.max += ela.max;

				countSum += count;
			}
			PCOUT(_finemat.GetPID(),"sum; " << elaSum.avg << "; " << elaSum.min << "; " << elaSum.max << "; " << countSum << std::endl)
		}

		OctreeGrid<T> &_finegrid;
		GenericMatrix<OctreeGrid<T> > &_finemat;

		int _smoothingsteps;
		int _level;
		double _w;

        DispVector *rFine;
        DispVector *eFine;
        DispVector *rCoarse;
        DispVector *eCoarse;

		MlOctreeGrid<T> * _coarsegrid;
		GenericMatrix<OctreeGrid<T> > * _coarsemat;
		Solver *_solver;
		Solver *presmoother;
		Solver *postsmoother;

		Timer *_timing;

		bool _ressourcesFreed;

		bool _first;
};

	template <class T>
MlCycle<T>::MlCycle(OctreeGrid<T> &finegrid, GenericMatrix<OctreeGrid<T> > &mat,Timer *timing,GPUGridBuilder<T> *gpuGridBuilder,GPUMemoryManager *gpuMM,bool switching,int degree, int level, int stype, double ratio, double w, SpectralOptions spectral): _finegrid(finegrid), _finemat(mat), _level(level), _w(w) {

	_timing = timing;
	Solver::_gpuMemoryManager = gpuMM;
	//Create coarser matrix and grid
	_coarsegrid = new MlOctreeGrid<T>(&finegrid,gpuGridBuilder,timing,level);
	_coarsegrid->GenerateBC();
	_coarsemat = new GenericMatrix<OctreeGrid<T> >(*_coarsegrid,level,_finemat._lMax,_finemat.GetTimer());
	_coarsemat->SetGPUMemoryManager(gpuMM);

	//Get local DOFS

	if (mat.GetPID() ==0) {
		std::cout << *_coarsemat;
	}
	t_coord ld[3];
	_coarsegrid->GetLocalDim(ld);
	_coarsegrid->GetGlobalDim(ld);
	SyncData *finestSyncData = 0;

	if (level == 0|| (ld[0] == 2) || (ld[1] == 2)|| (ld[2] == 2)) {
		long numBytes = _coarsegrid->GetNumBytes();
		cudaDeviceProp deviceProp;
		CCS(cudaGetDeviceProperties(&deviceProp,0),"MlCylce 1");
		if (switching) {
			_coarsegrid->ShiftDataToGPU(false,true);
		} else {
			if ((long) deviceProp.totalGlobalMem > numBytes) {
				_coarsegrid->ShiftDataToGPU(false,false);
			} else {
				_coarsegrid->ShiftDataToGPU(true,false);
			}
		}
		finestSyncData = _finegrid.GetFinestSyncData();
		if (!finestSyncData->switching) {
			CCS(cudaMalloc((void**) &finestSyncData->gVec,finestSyncData->gpuVecSize*sizeof(t_double)),"MlCycle 1");
		}
		_solver = new MlLevelCG(*_coarsemat,gpuMM);
	} else {
		_solver = new MlCycle(*_coarsegrid,*_coarsemat,_timing,gpuGridBuilder,gpuMM,switching,degree,level - 1,stype,ratio,w,spectral);
	}

	if (finestSyncData == 0) {
		finestSyncData = _finegrid.GetFinestSyncData();
	}
	if (finestSyncData->switching) {
		finestSyncData = 0;
	}

	if (_finemat.GetSyncData().switching) {
		rFine = new DispVectorS(_finemat.GetSyncData(),_gpuMemoryManager);
		eFine = new DispVectorS(_finemat.GetSyncData(),_gpuMemoryManager);
	} else if (finestSyncData != 0) {
		if (_finegrid.IsFinestGrid()) {
			rFine = new DispVectorN(_finemat.GetSyncData(),_gpuMemoryManager,finestSyncData->gVec);
			DispVectorN *rFineN = (DispVectorN*) rFine;
			rFineN->_sharedVec = false;
			_gpuMemoryManager->AllocateMemory(DispVec,finestSyncData->level,finestSyncData->gpuVecSize*sizeof(t_double));
			eFine = new DispVectorN(_finemat.GetSyncData(),_gpuMemoryManager);
		} else {
			rFine = new DispVectorN(_finemat.GetSyncData(),_gpuMemoryManager,finestSyncData->gVec);
			eFine = new DispVectorN(_finemat.GetSyncData(),_gpuMemoryManager,finestSyncData->gVec + _finemat.GetSyncData().gpuVecSize);
		}
	} else {
		rFine = new DispVectorN(_finemat.GetSyncData(),_gpuMemoryManager,_finemat.GetSyncData().gVec);
		eFine = new DispVectorN(_finemat.GetSyncData(),_gpuMemoryManager,_finemat.GetSyncData().gVec + _finemat.GetSyncData().gpuVecSize);
	}
	if (finestSyncData != 0) {
		rCoarse = new DispVectorN(_coarsemat->GetSyncData(),_gpuMemoryManager,finestSyncData->gVec + 2*_coarsemat->GetSyncData().gpuVecSize);
		eCoarse = new DispVectorN(_coarsemat->GetSyncData(),_gpuMemoryManager,finestSyncData->gVec + 3*_coarsemat->GetSyncData().gpuVecSize);
	} else {
		rCoarse = new DispVectorN(_coarsemat->GetSyncData(),_gpuMemoryManager,_finemat.GetSyncData().gVec + 2*_coarsemat->GetSyncData().gpuVecSize);
		eCoarse = new DispVectorN(_coarsemat->GetSyncData(),_gpuMemoryManager,_finemat.GetSyncData().gVec + 3*_coarsemat->GetSyncData().gpuVecSize);
	}

	rFine->Memset(0.0);
	eFine->Memset(0.0);
	rCoarse->Memset(0.0);
	eCoarse->Memset(0.0);

	// setting up cheb
	int MaxIters = spectral.effectiveLimit();

	if (stype ==0) {
		Chebyshev *tmpsmoother = new Chebyshev(_finemat,degree,true,*rFine,*eFine,gpuMM,MaxIters,0,0,ratio,spectral.mode == SpectralMode::Adaptive, spectral.levelDiagnostics.get(), level);
		presmoother = tmpsmoother;
		postsmoother = new Chebyshev(*tmpsmoother, false);
	} else {
		JacobiSmoother *tmpsmoother = new JacobiSmoother(_finemat,degree + level,w,true,*rFine);
		presmoother = tmpsmoother;
		postsmoother = new JacobiSmoother(*tmpsmoother, false);
	}
	_ressourcesFreed = false;
	_first = true;
	return;
}


template <class T>
int MlCycle<T>::Solve(DispVector &b,DispVector &x,cudaStream_t streamB,cudaStream_t streamX) {

	CCS(cudaStreamSynchronize(streamB),"Solve 1");
	CCS(cudaStreamSynchronize(streamX),"Solve 2");

	presmoother->Solve(b,x,streamB,streamX);

	// W cycle ( cy == 2)
	for (int cy = 0; cy < 2; cy++) {
		_finemat.Apply(x,*eFine,streamB,streamX);
		SubtractVector2(b,*eFine);

		cudaEvent_t startr;
		cudaEvent_t stopr;
		CCS(cudaEventCreate(&startr),"Solve 3");
		CCS(cudaEventCreate(&stopr),"Solve 4");
		CCS(cudaEventRecord(startr,0),"Solve 5");

		_coarsegrid->Restrict(*eFine,*rCoarse,streamB,streamX);

		CCS(cudaEventRecord(stopr,0),"Solve 6");
		CCS(cudaEventSynchronize(stopr),"Solve 7");
		float ela;
		CCS(cudaEventElapsedTime(&ela,startr,stopr),"Solve 8");
		_timing->Add(1,_level,ela/1000);

		eCoarse->Memset(0.0);

		//Solve the coarse size problem
		_solver->Solve(*rCoarse,*eCoarse,streamB,streamX);

		cudaEvent_t startp;
		cudaEvent_t stopp;
		CCS(cudaEventCreate(&startp),"Solve 9");
		CCS(cudaEventCreate(&stopp),"Solve 10");
		CCS(cudaEventRecord(startp,0),"Solve 11");

		_coarsegrid->Prolongate(*eCoarse,*eFine,streamB,streamX);

		CCS(cudaEventRecord(stopp,0),"Solve 12");
		CCS(cudaEventSynchronize(stopp),"Solve 13");
		CCS(cudaEventElapsedTime(&ela,startp,stopp),"Solve 14");
		_timing->Add(2,_level,ela/1000);
		
		AddVector(x,*eFine);

		postsmoother->Solve(b,x,streamB,streamX);

	}

	_finemat.GenerateDiagonal();
	if (!_finemat.dia->_nonZero) {
		ADiaK(x,*_finemat.dia);
	}

	CCS(cudaStreamSynchronize(streamB),"Solve 15");
	CCS(cudaStreamSynchronize(streamX),"Solve 16");

	return 1;
} 
#endif /* FULLGRIDMLCYCLE_H */

