/*
 * DiagonalS.cpp
 *
 * GParOSol: a parallel FE solver for trabecular bone modeling
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

#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include "DiagonalS.h"
#include "VectorKernels.h"
#include "OtherKernels.h"

DiagonalS::DiagonalS(SyncData &syncData,GPUMemoryManager *gpuMM) : Diagonal(syncData,gpuMM) {
	_cpuVecSize = Diagonal::_syncData.gpuVecSize - Diagonal::_syncData.tailSize*3;
	CCS(cudaHostAlloc((void**) &cpuVec,_cpuVecSize*sizeof(t_double),cudaHostAllocDefault),"PCGSlverMC 1");
}

DiagonalS::~DiagonalS() {
	CCS(cudaFreeHost(cpuVec),"~PCGSolverMC 1");
}

t_double* DiagonalS::GetVector() {
	return Diagonal::_syncData.gpuVecA;
}

void DiagonalS::Memset(t_double val,cudaStream_t stream) {
	for (uint i = 0;i < _cpuVecSize;i++) {
		cpuVec[i] = val;
	}
}

void DiagonalS::PrepareIn(cudaStream_t stream) {
	MemsetK(Diagonal::_syncData.gpuVecSize,Diagonal::_syncData.gpuVecA,0.0,stream);
}

void DiagonalS::PostProcess(cudaStream_t stream) {
	for (uint i = 0;i < Diagonal::_syncData.bufferSize;i++) {
		cpuVec[Diagonal::_syncData.cpuInds[i]*3] += _syncData.cpuBuffer[i*3];
		cpuVec[Diagonal::_syncData.cpuInds[i]*3 + 1] += _syncData.cpuBuffer[i*3 + 1];
		cpuVec[Diagonal::_syncData.cpuInds[i]*3 + 2] += _syncData.cpuBuffer[i*3 + 2];
	}
}

void DiagonalS::CopyBufferToGpu() {

}

void DiagonalS::CopyTailToCpu() {
	CCS(cudaMemcpy(Diagonal::_syncData.cpuBuffer + Diagonal::_syncData.bufferSize*3,Diagonal::_syncData.gpuVecA + _cpuVecSize,Diagonal::_syncData.tailSize*3*sizeof(t_double),cudaMemcpyDeviceToHost),"CopyTailToCpu 1");
	CCS(cudaMemcpy(cpuVec,Diagonal::_syncData.gpuVecA,_cpuVecSize*sizeof(t_double),cudaMemcpyDeviceToHost),"CopyTailToCpu 2");
}
