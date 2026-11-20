/*
 * DispVectorS.cpp
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
#include "DispVectorS.h"
#include "VectorKernels.h"
#include "OtherKernels.h"

DispVectorS::DispVectorS(SyncData &syncData,GPUMemoryManager *gpuMM) : DispVector(syncData,gpuMM) {
	_cpuVecSize = DispVector::_syncData.gpuVecSize - DispVector::_syncData.tailSize*3;
	CCS(cudaHostAlloc((void**) &cpuVec,_cpuVecSize*sizeof(t_double),cudaHostAllocDefault),"PCGSlverMC 1");
}

DispVectorS::~DispVectorS() {
	CCS(cudaFreeHost(cpuVec),"~PCGSolverMC 1");
}

t_double* DispVectorS::GetVector() {
	if (_vecIn) {
		return DispVector::_syncData.gpuVecA;
	} else {
		return DispVector::_syncData.gpuVecB;
	}
}

void DispVectorS::Memset(t_double val,cudaStream_t stream) {
	for (uint i = 0;i < _cpuVecSize;i++) {
		cpuVec[i] = val;
	}
}

void DispVectorS::Copy(DispVector &vec) {
	DispVectorS *dVec = (DispVectorS*) &vec;
	memcpy(cpuVec,dVec->cpuVec,_cpuVecSize*sizeof(t_double));
}

void DispVectorS::CopyToExtCpuVector(t_double *vec) {
	memcpy(vec,cpuVec,_cpuVecSize*sizeof(t_double));
}

void DispVectorS::CopyFromExtCpuVector(t_double *vec) {
	memcpy(cpuVec,vec,_cpuVecSize*sizeof(t_double));
}

void DispVectorS::CopyToExtCpuVectorAsync(t_double *vec,cudaStream_t stream) {
	CopyToExtCpuVector(vec);
}

void DispVectorS::CopyFromExtCpuVectorAsync(t_double *vec,cudaStream_t stream) {
	CopyFromExtCpuVector(vec);
}

void DispVectorS::PrepareIn(cudaStream_t stream) {
	_vecIn = true;
	CCS(cudaMemcpy(DispVector::_syncData.gpuVecA,cpuVec,_cpuVecSize*sizeof(t_double),cudaMemcpyHostToDevice),"CopyFromExtCpuVector 1");
	for (uint i = 0;i < DispVector::_syncData.bufferSize;i++) {
		DispVector::_syncData.cpuBuffer[i*3] = cpuVec[DispVector::_syncData.cpuInds[i]*3];
		DispVector::_syncData.cpuBuffer[i*3 + 1] = cpuVec[DispVector::_syncData.cpuInds[i]*3 + 1];
		DispVector::_syncData.cpuBuffer[i*3 + 2] = cpuVec[DispVector::_syncData.cpuInds[i]*3 + 2];
	}
}

void DispVectorS::PrepareOut(cudaStream_t stream,bool vecIn) {
	_vecIn = vecIn;
	if (_vecIn) {
		MemsetK(DispVector::_syncData.gpuVecSize,DispVector::_syncData.gpuVecA,0.0,stream);
	} else {
		MemsetK(DispVector::_syncData.gpuVecSize,DispVector::_syncData.gpuVecB,0.0,stream);
	}
}

void DispVectorS::PostProcess(cudaStream_t stream) {
	for (uint i = 0;i < DispVector::_syncData.bufferSize;i++) {
		cpuVec[DispVector::_syncData.cpuInds[i]*3] += _syncData.cpuBuffer[i*3];
		cpuVec[DispVector::_syncData.cpuInds[i]*3 + 1] += _syncData.cpuBuffer[i*3 + 1];
		cpuVec[DispVector::_syncData.cpuInds[i]*3 + 2] += _syncData.cpuBuffer[i*3 + 2];
	}
}

void DispVectorS::CopyTailToCpu() {
	if (_vecIn) {
		CCS(cudaMemcpy(DispVector::_syncData.cpuBuffer + DispVector::_syncData.bufferSize*3,DispVector::_syncData.gpuVecA + _cpuVecSize,DispVector::_syncData.tailSize*3*sizeof(t_double),cudaMemcpyDeviceToHost),"CopyTailToCpu 1");
		CCS(cudaMemcpy(cpuVec,DispVector::_syncData.gpuVecA,_cpuVecSize*sizeof(t_double),cudaMemcpyDeviceToHost),"CopyTailToCpu 2");
	} else {
		CCS(cudaMemcpy(DispVector::_syncData.cpuBuffer + DispVector::_syncData.bufferSize*3,DispVector::_syncData.gpuVecB + _cpuVecSize,DispVector::_syncData.tailSize*3*sizeof(t_double),cudaMemcpyDeviceToHost),"CopyTailToCpu 1");
		CCS(cudaMemcpy(cpuVec,DispVector::_syncData.gpuVecB,_cpuVecSize*sizeof(t_double),cudaMemcpyDeviceToHost),"CopyTailToCpu 2");
	}
}

void DispVectorS::CopyTailToGpu() {
	CCS(cudaMemcpy(DispVector::_syncData.gpuVecA + _cpuVecSize,DispVector::_syncData.cpuBuffer + DispVector::_syncData.bufferSize*3,DispVector::_syncData.tailSize*3*sizeof(t_double),cudaMemcpyHostToDevice),"CopyTailToGpu 1");
}
