/*
 * DiagnoalN.cpp
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
#include "DiagonalN.h"
#include "VectorKernels.h"
#include "OtherKernels.h"

DiagonalN::DiagonalN(SyncData &syncData,GPUMemoryManager *gpuMM) : Diagonal(syncData,gpuMM) {
	_gpuVecSize = Diagonal::_syncData.gpuVecSize;
	_gpuMM->AllocateMemory(Diag,Diagonal::_syncData.level,_gpuVecSize*sizeof(t_double));
	CCS(cudaMalloc((void**) &gpuVec,_gpuVecSize*sizeof(t_double)),"DispVector 1");
}

DiagonalN::~DiagonalN() {
	_gpuMM->FreeMemory(Diag,Diagonal::_syncData.level);
	CCS(cudaFree(gpuVec),"~DispVector 1");
}

t_double* DiagonalN::GetVector() {
	return gpuVec;
}

void DiagonalN::Memset(t_double val,cudaStream_t stream) {
	MemsetK(_gpuVecSize,gpuVec,val,stream);
}

void DiagonalN::PrepareIn(cudaStream_t stream) {
	Memset(0,stream);
}

void DiagonalN::PostProcess(cudaStream_t stream) {
	CopyBufferToGpu();
	AddInBufferK(Diagonal::_syncData.bufferSize,Diagonal::_syncData.gpuInds,gpuVec,Diagonal::_syncData.gpuBuffer);
}

void DiagonalN::CopyBufferToGpu() {
	CCS(cudaMemcpy(_syncData.gpuBuffer,_syncData.cpuBuffer,_syncData.bufferSize*3*sizeof(t_double),cudaMemcpyHostToDevice),"CopyBufferToGpu 1");
}

void DiagonalN::CopyTailToCpu() {
	CCS(cudaMemcpy(_syncData.cpuBuffer + _syncData.bufferSize*3,gpuVec + _gpuVecSize - _syncData.tailSize*3,_syncData.tailSize*3*sizeof(t_double),cudaMemcpyDeviceToHost),"CopyTailToCpu 1");
}
