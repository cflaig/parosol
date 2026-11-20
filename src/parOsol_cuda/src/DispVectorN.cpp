/*
 * DispVectorN.cpp
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
#include "DispVectorN.h"
#include "VectorKernels.h"
#include "OtherKernels.h"

DispVectorN::DispVectorN(SyncData &syncData,GPUMemoryManager *gpuMM) : DispVector(syncData,gpuMM) {
	_gpuVecSize = DispVector::_syncData.gpuVecSize;
	_sharedVec = false;
	_gpuMM->AllocateMemory(DispVec,DispVector::_syncData.level,_gpuVecSize*sizeof(t_double));
	CCS(cudaMalloc((void**) &gpuVec,_gpuVecSize*sizeof(t_double)),"DispVector 1");

}

DispVectorN::DispVectorN(SyncData &syncData,GPUMemoryManager *gpuMM,t_double *gVec) : DispVector(syncData,gpuMM) {
	_gpuVecSize = DispVector::_syncData.gpuVecSize;
	_sharedVec = true;
	gpuVec = gVec;
}

DispVectorN::~DispVectorN() {
	if (!_sharedVec) {
		_gpuMM->FreeMemory(DispVec,DispVector::_syncData.level);
		CCS(cudaFree(gpuVec),"~DispVector 1");
	}
}

t_double* DispVectorN::GetVector() {
	return gpuVec;
}

void DispVectorN::Memset(t_double val,cudaStream_t stream) {
	MemsetK(_gpuVecSize,gpuVec,val,stream);
}

void DispVectorN::Copy(DispVector &vec) {
	CCS(cudaMemcpy(gpuVec,vec.GetVector(),_gpuVecSize*sizeof(t_double),cudaMemcpyDeviceToDevice),"Copy 1");
}

void DispVectorN::CopyToExtCpuVector(t_double *vec) {
	CCS(cudaMemcpy(vec,gpuVec,_gpuVecSize*sizeof(t_double),cudaMemcpyDeviceToHost),"CopyToExtCpuVector 1");
}

void DispVectorN::CopyFromExtCpuVector(t_double *vec) {
	CCS(cudaMemcpy(gpuVec,vec,_gpuVecSize*sizeof(t_double),cudaMemcpyHostToDevice),"CopyFromExtCpuVector 1");
}

void DispVectorN::CopyToExtCpuVectorAsync(t_double *vec,cudaStream_t stream) {
	CCS(cudaMemcpyAsync(vec,gpuVec,_gpuVecSize*sizeof(t_double),cudaMemcpyDeviceToHost,stream),"CopyToExtCpuVector 1");
}

void DispVectorN::CopyFromExtCpuVectorAsync(t_double *vec,cudaStream_t stream) {
	CCS(cudaMemcpyAsync(gpuVec,vec,_gpuVecSize*sizeof(t_double),cudaMemcpyHostToDevice,stream),"CopyFromExtCpuVector 1");
}

void DispVectorN::PrepareIn(cudaStream_t stream) {
		GatherSendBufferK(DispVector::_syncData.bufferSize,DispVector::_syncData.gpuInds,gpuVec,DispVector::_syncData.gpuBuffer);
		CopyBufferToCpu();
}

void DispVectorN::PrepareOut(cudaStream_t stream,bool vecIn) {
	Memset(0,stream);
}

void DispVectorN::PostProcess(cudaStream_t stream) {
	CopyBufferToGpu();
	AddInBufferK(DispVector::_syncData.bufferSize,DispVector::_syncData.gpuInds,gpuVec,DispVector::_syncData.gpuBuffer);
}

void DispVectorN::CopyBufferToCpu() {
	CCS(cudaMemcpy(_syncData.cpuBuffer,_syncData.gpuBuffer,_syncData.bufferSize*3*sizeof(t_double),cudaMemcpyDeviceToHost),"CopyBufferToCpu 1");
}

void DispVectorN::CopyBufferToGpu() {
	CCS(cudaMemcpy(_syncData.gpuBuffer,_syncData.cpuBuffer,_syncData.bufferSize*3*sizeof(t_double),cudaMemcpyHostToDevice),"CopyBufferToGpu 1");
}

void DispVectorN::CopyTailToCpu() {
	CCS(cudaMemcpy(_syncData.cpuBuffer + _syncData.bufferSize*3,gpuVec + _gpuVecSize - _syncData.tailSize*3,_syncData.tailSize*3*sizeof(t_double),cudaMemcpyDeviceToHost),"CopyTailToCpu 1");
}

void DispVectorN::CopyTailToGpu() {
	CCS(cudaMemcpy(gpuVec + _gpuVecSize - _syncData.tailSize*3,_syncData.cpuBuffer + _syncData.bufferSize*3,_syncData.tailSize*3*sizeof(t_double),cudaMemcpyHostToDevice),"CopyTailToGpu 1");
}
