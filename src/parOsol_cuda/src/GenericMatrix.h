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


#ifndef GENERICMATRIX_H
#define GENERICMATRIX_H

#include <iomanip>
#include <cuda_runtime.h>
#include "Config.h"
#include "Timing.h"
#include "StiffnessMatrix.h"
#include "SpectralOptions.h"
#include <vector>
#include "BoundaryCondition.h"
#include "Toolbox.h"
#include "fem.h"
#include "OptLocalMatrix.h"
#include "ApplyKernels.h"
#include "VectorOperations.h"
#include "OtherKernels.h"
#include "DispVectorN.h"
#include "Diagonal.h"
#include "DiagonalS.h"
#include "DiagonalN.h"
#include "DiagonalNComp.h"
#include "GPUMemoryManager.h"

void decarbenz(double  *Y, double  *X);

//! A class to perform an matrix-vector product for elasticity problem based on grid

/*! The GenericMatrix class is used to multiply a vector by the global stiffness
 * matrix, whithout ever assembling the matrix. A local stiffness matrix is
 * computed. This is applied element wise to the nodes corresponding with the
 * weight of the bone. The base grid is given by a template parameter
 */

template <class Grid>
class GenericMatrix: public StiffnessMatrix
{
	public:
		GenericMatrix(Grid &grid,uint level,uint lMax,Timer *timing);

		~GenericMatrix();

		/** The Application is done with respect to the boundary condition. The matrix is not assembled
		 *  This function needs communication.
		 *
		 *  @param [in] x input vector
		 *  @param [out] y output vector
		 */
		int  Apply (DispVector &x,DispVector &y,cudaStream_t streamX = 0,cudaStream_t streamY = 0);
		int  ApplyCheb (bool first,bool last,DispVector &x,DispVector &y,t_double alpha,t_double beta,Diagonal &a,DispVector &b,DispVector &d,cudaStream_t streamX = 0,cudaStream_t streamY = 0);
		int  ApplyCheb2 (bool first,bool last,DispVector &x,DispVector &y,t_double alpha,t_double beta,Diagonal &a,DispVector &b,DispVector &d,cudaStream_t streamX = 0,cudaStream_t streamY = 0);

		/** This application of the matrix will not apply the boundary condition. It uses the indefinite matrix.
		 *  This function needs communication.
		 *
		 *  @param [in] x input vector
		 *  @param [out] y output vector
		 */
		int Apply_NoResetBoundaries(DispVector &x,DispVector &y,cudaStream_t streamX = 0,cudaStream_t streamY = 0);

		inline virtual const std::string  Label () const {
			return("Matrix");
		}

		/// Prepares the data structure that is needed by apply
		int PrepareApply();

        void GenerateDiagonal();

		Grid& GetGrid() {
			return _grid;
		}

		BoundaryCondition & GetBC()
		{
			return _grid.bc;
		}

		
		//!Get the local Number of Dofs  
		t_index GetNrDofs()
		{
			return _mydofs;
		}

		/** Computes the dot product of the vectors based on the grid
		 *  @param a first input vector
		 *  @param b second input vector
		 */
		t_double dot(DispVector &a,DispVector &b,cudaStream_t stream = 0) {
			return _grid.dot(a,b,stream);
		}

		t_double dot(DispVector &a,cudaStream_t stream = 0) {
			return _grid.dot(a,stream);
		}

		void ddot(DispVector &a,DispVector &b,t_double &dot1,t_double &dot2,cudaStream_t stream = 0) {
			_grid.ddot(a,b,dot1,dot2,stream);
		}

		virtual SyncData& GetSyncData() {
			return _grid._gpuGrid->_syncData;
		}


		friend std::ostream& operator<<(std::ostream& stream, const GenericMatrix &mat)
		{
			return mat.print_prop(stream);
		}

		Timer* GetTimer() {
			return _timing;
		}

		int GetPID() {
			return MyPID;
		}

		// Build the same reproducible node-key start as the CPU fork. The
		// host vector is temporary; ghost values are synchronized before upload.
		void SetVectorRandom(DispVector &x) {
			std::vector<t_double> values(GetNrDofs());
			const auto &nodes = _grid.GetOctGrid();
			for (t_index i=0; i<GetNrDofs()/3; ++i)
				for (unsigned c=0; c<3; ++c)
					values[3*i+c] = SpectralRandomValue(std::uint64_t(nodes[i].key), c);
			_grid.Recv_import_Ghost(values.data());
			_grid.Send_import_Ghost(values.data());
			_grid.Wait_import_Ghost();
			x.CopyFromExtCpuVector(values.data());
		}

		void PrintTimings() {
			t_timing elaSum;
			elaSum.min = 0;
			elaSum.avg = 0;
			elaSum.max = 0;
			t_timing elaOSum;
			elaOSum.min = 0;
			elaOSum.avg = 0;
			elaOSum.max = 0;
			PCOUT(MyPID,"Apply" << std::endl)
			PCOUT(MyPID,"level; time avg; time min; time max; calls; overlap avg; overlap min; overlap max" << std::endl)
			COUTPRECISION(5);
			int countSum = 0;
			for (int i = _lMax;i >= 0;i--) {
				std::string str = "Apply ";

				t_timing ela = _timing->ElapsedTime(0,i);
				t_timing elaO = _timing->ElapsedTime(3,i);
				int count = _timing->Counter(0,i);
				PCOUT(MyPID,i << "; " << ela.avg << "; " << ela.min << "; " << ela.max << "; " << count << "; " << elaO.avg << "; " << elaO.min << "; " << elaO.max << "\n")
				elaSum.min += ela.min;
				elaSum.avg += ela.avg;
				elaSum.max += ela.max;
				elaOSum.min += elaO.min;
				elaOSum.avg += elaO.avg;
				elaOSum.max += elaO.max;

				countSum += count;
			}
			PCOUT(MyPID,"sum; " << elaSum.avg << "; " << elaSum.min << "; " << elaSum.max << "; " << countSum << "; " << elaOSum.avg << "; " << elaOSum.min << "; " << elaOSum.max << "\n")
			COUTPRECISION(13);
		}

		void SetGPUMemoryManager(GPUMemoryManager *gpuMM) {
			_gpuMemoryManager = gpuMM;
		}

		GPUMemoryManager *_gpuMemoryManager;

		uint _level;
		uint _lMax;

	private:
		void ResetBoundaries(DispVector &y,cudaStream_t stream = 0);
		void SetBoundaries(DispVector &y,DispVector &x,cudaStream_t = 0) const;

        t_double  *_stiffnessmatrix;
		double* _matprop;
		t_index _mydofs;
		Grid& _grid;

		double _factor;

		t_double *dev_stiffnessmatrix;

		//some print function
		std::ostream& print_mat(std::ostream& stream) const;
		std::ostream& print_prop(std::ostream& stream) const;

		int MyPID;
		Timer *_timing;

};

template <class Grid>
GenericMatrix<Grid>::GenericMatrix(Grid &grid,uint level,uint lMax,Timer *timing) : _grid(grid) {
	_gpuMemoryManager = new GPUMemoryManager();
	_level = level;
	_lMax = lMax;
	_timing = timing;
	dia = 0;
	MyPID = _grid.GetPID();

	_mydofs = grid.GetNrDofs();

	//set up the local stiffness matrix
	double GridDim[3];
	int Dimension = 3;
	const int NumMaterialProps = 2;
	_matprop = new double[2];
	_matprop[0] = 1000; //refernece value Emodule is linear
	_matprop[1] = grid.poisons_ratio;
	int NumNodesPerElement = 8;
	int NumDofsPerElement = 24;
	int NumIntegrationPoints = 8;
	int SSMatrixSize = 6;
	double *coord = new double[Dimension * NumNodesPerElement];
	grid.GetRes(GridDim);
	setcoord(GridDim,coord);

	_stiffnessmatrix = new t_double[24*24];
	double *tmpstiff = new double[24*24];

	Stiffness_Matrix(_matprop, NumMaterialProps,
			NumNodesPerElement, NumDofsPerElement,
			Dimension, NumIntegrationPoints,
			SSMatrixSize, coord, tmpstiff); //TODO: not sure if this works

	for( int i = 0; i < 24*24;i++) {
		_stiffnessmatrix[i] = tmpstiff[i];
	}

	if (_lMax - _level < 12) {
		SetLocalStiffnessMatrix(tmpstiff,(_lMax - _level)*24*24);
	} else {
		_gpuMemoryManager->AllocateMemory(StiffMat,_level,24*24*sizeof(t_double));
		CCS(cudaMalloc((void**) &dev_stiffnessmatrix,24*24*sizeof(t_double)),"GenericMatrix 1");
		CCS(cudaMemcpy(dev_stiffnessmatrix,tmpstiff,24*24*sizeof(t_double),cudaMemcpyHostToDevice),"GenericMatrix 2");
	}

	delete[] coord;
	delete[] tmpstiff;
	const double nu = _matprop[1];
	//if the optimized code is used, following factor has to be used. Else 1
	_factor = (1-nu)/((1+nu)*(1-2*nu))*1000/(144*(1-nu))*GridDim[0];
	//_factor = 1;

	PrepareApply();

}

template <class Grid>
GenericMatrix<Grid>::~GenericMatrix() {
	delete[] _stiffnessmatrix;
	delete[] _matprop;
	if (dia) {
		delete dia;
	}
	if (_lMax - _level >= 12) {
		_gpuMemoryManager->FreeMemory(StiffMat,_level);
		CCS(cudaFree(dev_stiffnessmatrix),"~GenericMatrix 2");
	}
}

template <class Grid>
int  GenericMatrix<Grid>::Apply(DispVector &x,DispVector &y,cudaStream_t streamX,cudaStream_t streamY) {
	//Store the values on boundary... replace it with 0

	CCS(cudaStreamSynchronize(streamX),"Apply 1");
	CCS(cudaStreamSynchronize(streamY),"Apply 2");

	cudaEvent_t start;
	cudaEvent_t stop;
	CCS(cudaEventCreate(&start),"Apply 3");
	CCS(cudaEventCreate(&stop),"Apply 4");
	CCS(cudaEventRecord(start,streamX),"Apply 5");

	ResetBoundaries(x,streamX);
	//Import needed degree of freedoms....

	//Apply the singular matrices
	Apply_NoResetBoundaries(x,y,streamX,streamY);

	//Write back the stored disp
	CCS(cudaStreamSynchronize(streamX),"Apply 6");
	SetBoundaries(x,y,streamY);
	CCS(cudaStreamSynchronize(streamY),"Apply 7");

	CCS(cudaEventRecord(stop,streamY),"Apply 8");
	CCS(cudaEventSynchronize(stop),"Apply 9");
	float ela;
	CCS(cudaEventElapsedTime(&ela,start,stop),"Apply_NoResetBoundaries 10");
	_timing->Add(0,_level,ela/1000);
	return 0;
}

template <class Grid>
int  GenericMatrix<Grid>::ApplyCheb(bool first,bool last,DispVector &x,DispVector &y,t_double alpha,t_double beta,Diagonal &a,DispVector &b,DispVector &d,cudaStream_t streamX,cudaStream_t streamY) {
	int numChunks = 8;
	CCS(cudaStreamSynchronize(streamX),"Apply 1");
	CCS(cudaStreamSynchronize(streamY),"Apply 2");

	cudaEvent_t start;
	cudaEvent_t stop;
	CCS(cudaEventCreate(&start),"Apply 3");
	CCS(cudaEventCreate(&stop),"Apply 4");
	CCS(cudaEventRecord(start,streamX),"Apply 5");

	DiagonalS *aS = (DiagonalS*) &a;
	DispVectorS *bS = (DispVectorS*) &b;
	DispVectorS *yS = (DispVectorS*) &y;
	DispVectorS *dS = (DispVectorS*) &d;
	DispVectorS *xS = (DispVectorS*) &x;


	//Apply the singular matrices

	if (first) {
		ResetBoundaries(x);
		x.PrepareIn(0);
	} else {
		for (uint i = 0;i < GetSyncData().bufferSize;i++) {
			GetSyncData().cpuBuffer[i*3] = xS->cpuVec[GetSyncData().cpuInds[i]*3];
			GetSyncData().cpuBuffer[i*3 + 1] = xS->cpuVec[GetSyncData().cpuInds[i]*3 + 1];
			GetSyncData().cpuBuffer[i*3 + 2] = xS->cpuVec[GetSyncData().cpuInds[i]*3 + 2];
		}
	}

	_grid.Recv_import_Ghost(x._syncData);
	_grid.Send_import_Ghost(x._syncData);
	y.PrepareOut(0);
	_grid.Wait_import_Ghost();

	x.CopyTailToGpu();

	if (_lMax - _level < 12) {
		_grid._gpuGrid->MatrixVectorCK(x.GetVector(),y.GetVector(),(_lMax - _level)*576);
		} else {
		_grid._gpuGrid->MatrixVectorK(x.GetVector(),y.GetVector(),dev_stiffnessmatrix);
	}

	CCS(cudaMemcpy(GetSyncData().cpuBuffer + GetSyncData().bufferSize*3,GetSyncData().gpuVecB + yS->_cpuVecSize,GetSyncData().tailSize*3*sizeof(t_double),cudaMemcpyDeviceToHost),"CopyTailToCpu 1");
	_grid.Recv_export_Ghost(y._syncData);
	_grid.Send_export_Ghost(y._syncData);

	uint chunkSize = yS->_cpuVecSize/numChunks;
	if (chunkSize % 4 != 0) {
		chunkSize += 4 - (chunkSize % 4);
	}
	uint lastChunk = yS->_cpuVecSize - (numChunks - 1)*chunkSize;

	CCS(cudaMemcpyAsync(yS->cpuVec,GetSyncData().gpuVecB,chunkSize*sizeof(t_double),cudaMemcpyDeviceToHost,streamX),"CopyTailToCpu 2");

	_grid.WaitAndCopy_export_Ghost();

	uint boundaryInd = 0;
	for (int chunk = 0;chunk < numChunks;chunk++) {
		CCS(cudaStreamSynchronize(streamX),"cheb 1");
		if (chunk < numChunks - 2) {
			CCS(cudaMemcpyAsync(yS->cpuVec + (chunk + 1)*chunkSize,GetSyncData().gpuVecB + (chunk + 1)*chunkSize,chunkSize*sizeof(t_double),cudaMemcpyDeviceToHost,streamX),"CopyTailToCpu 2");
		} else if (chunk == numChunks - 2) {
			CCS(cudaMemcpyAsync(yS->cpuVec + (chunk + 1)*chunkSize,GetSyncData().gpuVecB + (chunk + 1)*chunkSize,lastChunk*sizeof(t_double),cudaMemcpyDeviceToHost,streamX),"CopyTailToCpu 2");
		}

		uint stt = chunk*chunkSize;
		uint end = stt + chunkSize;
		if (chunk == numChunks - 1) {
			end = stt + lastChunk;
		}

		for (uint i = 0;i < GetSyncData().bufferSize;i++) {
			for (int j = 0;j < 3;j++) {
				if (GetSyncData().cpuInds[i]*3 + j >= stt && GetSyncData().cpuInds[i]*3 + j < end) {
					yS->cpuVec[GetSyncData().cpuInds[i]*3 + j] += GetSyncData().cpuBuffer[i*3 + j];
				}
			}
		}

		uint tBoundaryInd = boundaryInd;
		while (_grid.bc.FixedNodesInd[boundaryInd] < end && boundaryInd < _grid.bc.FixedNodesInd.size()) {
			xS->cpuVec[_grid.bc.FixedNodesInd[boundaryInd]] = _grid.bc.storedDisp[boundaryInd];
			yS->cpuVec[_grid.bc.FixedNodesInd[boundaryInd]] = _grid.bc.storedDisp[boundaryInd];
			boundaryInd++;
		}

#pragma omp parallel for
#ifdef MMX
		for (uint i = stt;i < end;i = i + 4) {
			__m256d amm = _mm256_load_pd(&aS->cpuVec[i]);
			__m256d bmm = _mm256_load_pd(&bS->cpuVec[i]);
			__m256d cmm = _mm256_load_pd(&yS->cpuVec[i]);
			__m256d dmm = _mm256_load_pd(&dS->cpuVec[i]);
			__m256d alphamm = _mm256_broadcast_sd(&alpha);
			__m256d betamm = _mm256_broadcast_sd(&beta);
			amm = _mm256_mul_pd(amm,alphamm);
			bmm = _mm256_sub_pd(bmm,cmm);
			amm = _mm256_mul_pd(amm,bmm);
			dmm = _mm256_mul_pd(dmm,betamm);
			dmm = _mm256_add_pd(dmm,amm);
			_mm256_store_pd(&dS->cpuVec[i],dmm);
			__m256d emm = _mm256_load_pd(&xS->cpuVec[i]);
			emm = _mm256_add_pd(emm,dmm);
			_mm256_store_pd(&xS->cpuVec[i],emm);
		}
#else
		for (uint i = stt;i < end;i++) {
			dS->cpuVec[i] = alpha*aS->cpuVec[i]*(bS->cpuVec[i] - yS->cpuVec[i]) + dS->cpuVec[i]*beta;
			xS->cpuVec[i] += dS->cpuVec[i];
		}
#endif

		if (!last) {
			while (_grid.bc.FixedNodesInd[tBoundaryInd] < end && tBoundaryInd < _grid.bc.FixedNodesInd.size()) {
				_grid.bc.storedDisp[tBoundaryInd] = xS->cpuVec[_grid.bc.FixedNodesInd[tBoundaryInd]];
				xS->cpuVec[_grid.bc.FixedNodesInd[tBoundaryInd]] = 0;
				tBoundaryInd++;
			}
			if (chunk < numChunks - 1) {
				CCS(cudaMemcpyAsync(GetSyncData().gpuVecA + chunk*chunkSize,xS->cpuVec + chunk*chunkSize,chunkSize*sizeof(t_double),cudaMemcpyHostToDevice,streamY),"CopyFromExtCpuVector 1");
			} else {
				CCS(cudaMemcpyAsync(GetSyncData().gpuVecA + chunk*chunkSize,xS->cpuVec + chunk*chunkSize,lastChunk*sizeof(t_double),cudaMemcpyHostToDevice,streamY),"CopyFromExtCpuVector 1");
			}
		}
	}

	CCS(cudaStreamSynchronize(streamY),"Apply 7");

	CCS(cudaEventRecord(stop,streamY),"Apply 8");
	CCS(cudaEventSynchronize(stop),"Apply 9");
	float ela;
	CCS(cudaEventElapsedTime(&ela,start,stop),"Apply_NoResetBoundaries 10");
	_timing->Add(0,_level,ela/1000);
	return 0;
}

template <class Grid>
int GenericMatrix<Grid>::Apply_NoResetBoundaries(DispVector &x,DispVector &y,cudaStream_t streamX,cudaStream_t streamY) {

	x.PrepareIn(0);

	_grid.Recv_import_Ghost(x._syncData);
	_grid.Send_import_Ghost(x._syncData);
	y.PrepareOut(0);
	_grid.Wait_import_Ghost();

	x.CopyTailToGpu();

	if (_lMax - _level < 12) {
		_grid._gpuGrid->MatrixVectorCK(x.GetVector(),y.GetVector(),(_lMax - _level)*576);
		} else {
		_grid._gpuGrid->MatrixVectorK(x.GetVector(),y.GetVector(),dev_stiffnessmatrix);
	}

	y.CopyTailToCpu();

	_grid.Recv_export_Ghost(y._syncData);
	_grid.Send_export_Ghost(y._syncData);
	_grid.WaitAndCopy_export_Ghost();

	y.PostProcess(0);

	return 0;
}

template <class Grid>
void GenericMatrix<Grid>::GenerateDiagonal() {
	if (dia == 0) {
		if (_grid._gpuGrid->_syncData.switching) {
			dia = new DiagonalS(_grid._gpuGrid->_syncData,_gpuMemoryManager);
		} else {
			if (GetSyncData().compressDia) {
				dia = new DiagonalNComp(GetSyncData(),_gpuMemoryManager,&_grid.bc);
			} else {
				dia = new DiagonalN(_grid._gpuGrid->_syncData,_gpuMemoryManager);
			}
		}

		_grid.Recv_export_Ghost(dia->_syncData);
		dia->PrepareIn(0);

		if (!GetSyncData().switching && GetSyncData().compressDia) {

			_grid._gpuGrid->MatrixVectorDiaCompK(dia->GetVector(),_stiffnessmatrix[0]);
			CCS(cudaDeviceSynchronize(),"hefaef1");
		} else {

			_grid._gpuGrid->MatrixVectorDiaK(dia->GetVector(),_stiffnessmatrix[0]);
			CCS(cudaDeviceSynchronize(),"hefaef2");
		}

		CCS(cudaDeviceSynchronize(),"hefaef");

		dia->CopyTailToCpu();

		_grid.Send_export_Ghost(dia->_syncData);
		_grid.WaitAndCopy_export_Ghost();

		dia->PostProcess(0);

		// apply fixed nodes on vec
		//1.0 for diagonal, 0.0 for rhs
		SetDiaK(_grid.bc,*dia);

		//compute the pseudo invers of the diagonal. If the entry is zero than set it also in
		//the inverse to zero
		bool hasZeros = PInverseDia(*dia,0);
		dia->_nonZero = !hasZeros;

	}

}

//Compute the indicies of the nodes affactet by the BC
template <class Grid>
int GenericMatrix<Grid>::PrepareApply() {
	return 0;
}
template <class Grid>
void GenericMatrix<Grid>::SetBoundaries(DispVector &y,DispVector &x,cudaStream_t stream) const {
	SetBoundariesK(_grid.bc,x,y,stream);
}

//! Sets all boundary nodes (both fixed and restrained) to zero.
template <class Grid>
void GenericMatrix<Grid>::ResetBoundaries(DispVector &y,cudaStream_t stream) {
	ResetBoundariesK(_grid.bc,y,stream);
}

template <class Grid>
std::ostream& GenericMatrix<Grid>::print_mat(std::ostream& stream) const {
	for (int i = 0;i < 24; i++) {
		for (int j = 0;j < 24; j++)
			stream << std::setiosflags(std::ios::fixed)<< std::setw( 7) << std::setprecision( 4 )<< _stiffnessmatrix[i*24 + j] << " ";
		stream << std::endl;
	}
	return stream;
}

template <class Grid>
std::ostream& GenericMatrix<Grid>::print_prop(std::ostream& stream) const {
	double GridDim[3];
	_grid.GetRes(GridDim);
	t_coord GlobalDim[3];
	_grid.GetGlobalDim(GlobalDim);
	stream << "Matrix:\n";
	stream << "   used Grid Size: " << GlobalDim[0] << ", " << GlobalDim[1] << ", " << GlobalDim[2]  << "\n";
	stream << "   used Grid Resolution: " << GridDim[0] << ", " << GridDim[1] << ", " << GridDim[2]  << "\n";
	stream << "   used Material properties:   E: " << _matprop[0] << " Poisson: "<< _matprop[1] << std::endl;
	return stream;
}

#endif /* GENERICMATRIX_H */


