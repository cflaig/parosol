/*
 * OtherKernels.cu
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

#define THREADS_PER_BLOCK 256

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "Config.h"
#include "OtherKernels.h"

__device__ t_double AtomicAdd2(t_double *address,t_double val) {
	unsigned long long int *address_as_ull = (unsigned long long int*) address;
	unsigned long long int old = *address_as_ull;
	unsigned long long int assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull,assumed,__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__global__ void MemsetKernel(t_index numStarts,t_double *a,t_double val) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	a[index] = val;
}

void MemsetK(t_index numStarts,t_double *dev_a,t_double val,cudaStream_t stream) {
	int blockcount = numStarts/THREADS_PER_BLOCK + 1;
	MemsetKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(numStarts,dev_a,val);
}

__global__ void ResetBoundariesKernel(t_index numStarts,const t_index *__restrict__ fixedNodes,t_double *stored_disp,t_double *y) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}

	stored_disp[index] = y[fixedNodes[index]];
	y[fixedNodes[index]] = 0;
}

void ResetBoundariesK(BoundaryCondition &bc,DispVector &y,cudaStream_t stream) {
	if (y._syncData.switching) {
		std::vector<t_index>::iterator iter;
		DispVectorS *yS = (DispVectorS*) &y;
		int i = 0;
		for (iter = bc.FixedNodesInd.begin();iter != bc.FixedNodesInd.end();iter++) {
			bc.storedDisp[i] = yS->cpuVec[*iter];
			yS->cpuVec[*iter] = 0;
			i++;
		}
	} else {
		int blockcount = bc.FixedNodesInd.size()/THREADS_PER_BLOCK + 1;
		ResetBoundariesKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(bc.FixedNodesInd.size(),bc.devFixedNodesInd,bc.devStoredDisp,y.GetVector());
	}
}

__global__ void SetBoundariesKernel(t_index numStarts,const t_index *__restrict__ fixedNodes,const t_double *__restrict__ stored_disp,t_double *x,t_double *y) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}

	x[fixedNodes[index]] = stored_disp[index];
	y[fixedNodes[index]] = x[fixedNodes[index]];
}

void SetBoundariesK(BoundaryCondition &bc,DispVector &x,DispVector &y,cudaStream_t stream) {
	if (y._syncData.switching) {
		std::vector<t_index>::iterator iter;
		DispVectorS *xS = (DispVectorS*) &x;
		DispVectorS *yS = (DispVectorS*) &y;
		int i = 0;
		for (iter = bc.FixedNodesInd.begin();iter != bc.FixedNodesInd.end();iter++) {
			xS->cpuVec[*iter] = bc.storedDisp[i];
			yS->cpuVec[*iter] = bc.storedDisp[i];
			i++;
		}
	} else {
		int blockcount = bc.FixedNodesInd.size()/THREADS_PER_BLOCK + 1;
		SetBoundariesKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(bc.FixedNodesInd.size(),bc.devFixedNodesInd,bc.devStoredDisp,x.GetVector(),y.GetVector());
	}
}

__global__ void SetDiaKernel(t_index numStarts,const t_index *__restrict__ fixedNodes,t_double *dia) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}

	dia[fixedNodes[index]] = 1.0;
}

void SetDiaK(BoundaryCondition &bc,Diagonal &dia,cudaStream_t stream) {
	if (dia._syncData.switching) {
		std::vector<t_index>::iterator iter;
		DiagonalS *diaS = (DiagonalS*) &dia;
		for (iter = bc.FixedNodesInd.begin();iter != bc.FixedNodesInd.end();iter++) {
			diaS->cpuVec[*iter] = 1.0;
		}
	} else {
		if (!dia._syncData.compressDia) {
			int blockcount = bc.FixedNodesInd.size()/THREADS_PER_BLOCK + 1;
			SetDiaKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(bc.FixedNodesInd.size(),bc.devFixedNodesInd,dia.GetVector());
		}
	}
}


__global__ void ADiaKernel(t_index numStarts,t_double *x,const t_double *__restrict__ dia) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	if (dia[index] == 0.0) {
		x[index] = 0.0;
	}
}

__global__ void ADiaCompKernel(t_index numStarts,const t_index *__restrict__ fixedInds,const t_double *__restrict__ x,t_double *store) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	store[index] = x[fixedInds[index]];
}

__global__ void ADiaComp2Kernel(t_index numStarts,t_double *x,const t_double *__restrict__ dia) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	if (dia[index] == 0.0) {
		x[index*3] = 0.0;
		x[index*3 + 1] = 0.0;
		x[index*3 + 2] = 0.0;
	}
}

__global__ void ADiaComp3Kernel(t_index numStarts,const t_index *__restrict__ fixedInds,t_double *x,const t_double *__restrict__ store) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	x[fixedInds[index]] = store[index];
}

void ADiaK(DispVector &x,Diagonal &dia,cudaStream_t stream) {
	if (x._syncData.switching) {
		DispVectorS *xS = (DispVectorS*) &x;
		DiagonalS *diaS = (DiagonalS*) &dia;
#pragma omp parallel for
		for (uint i = 0;i < xS->_cpuVecSize;i++) {
			if (diaS->cpuVec[i] == 0.0) {
				xS->cpuVec[i] = 0.0;
			}
		}
	} else {
		if (dia._syncData.compressDia) {
			DiagonalNComp *nDia = (DiagonalNComp*) &dia;
			int blockcount = nDia->_bc->FixedNodesInd.size()/THREADS_PER_BLOCK + 1;
			ADiaCompKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(nDia->_bc->FixedNodesInd.size(),nDia->_bc->devFixedNodesInd,x.GetVector(),nDia->_bc->devStoredDisp);
			int blockcount2 = nDia->_gpuVecSize/THREADS_PER_BLOCK + 1;
			ADiaComp2Kernel<<<blockcount2,THREADS_PER_BLOCK,0,stream>>>(nDia->_gpuVecSize,x.GetVector(),dia.GetVector());
			ADiaComp3Kernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(nDia->_bc->FixedNodesInd.size(),nDia->_bc->devFixedNodesInd,x.GetVector(),nDia->_bc->devStoredDisp);
		} else {
			int blockcount = x._syncData.gpuVecSize/THREADS_PER_BLOCK + 1;
			ADiaKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(x._syncData.gpuVecSize,x.GetVector(),dia.GetVector());
		}
	}
}

__global__ void SetBoundaries2Kernel(t_index numStarts,const t_index *__restrict__ fixedNodes,t_double *y) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}

	y[fixedNodes[index]] = 0;
}

void SetBoundaries2K(BoundaryCondition &bc,DispVector &y,cudaStream_t stream) {
	if (y._syncData.switching) {
		std::vector<t_index>::iterator iter;
		DispVectorS *yS = (DispVectorS*) &y;
		for (iter = bc.FixedNodesInd.begin();iter != bc.FixedNodesInd.end();iter++) {
			yS->cpuVec[*iter] = 0;
		}
	} else {
		int blockcount = bc.FixedNodesInd.size()/THREADS_PER_BLOCK + 1;
		SetBoundaries2Kernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(bc.FixedNodesInd.size(),bc.devFixedNodesInd,y.GetVector());
	}
}

__global__ void SetBoundaries3Kernel(t_index numStarts,const t_index *__restrict__ fixedNodes,const t_double *__restrict__ stored_disp,t_double *y) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	y[fixedNodes[index]] = stored_disp[index];
}

void SetBoundaries3K(BoundaryCondition &bc,DispVector &y,cudaStream_t stream) {
	if (y._syncData.switching) {
		std::vector<t_index>::iterator iter;
		DispVectorS *yS = (DispVectorS*) &y;
		int i = 0;
		for (iter = bc.FixedNodesInd.begin();iter != bc.FixedNodesInd.end();iter++) {
			yS->cpuVec[*iter] = bc.FixedNodes[i];
			i++;
		}
	} else {
		int blockcount = bc.FixedNodesInd.size()/THREADS_PER_BLOCK + 1;
		SetBoundaries3Kernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(bc.FixedNodesInd.size(),bc.devFixedNodesInd,bc.devFixedNodes,y.GetVector());
	}
}

__global__ void AddBoundariesKernel(t_index numStarts,const t_index *__restrict__ fixedNodes,const t_double *__restrict__ stored_disp,t_double *y) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	y[fixedNodes[index]] += stored_disp[index];
}

void AddBoundariesK(BoundaryCondition &bc,DispVector &y,cudaStream_t stream) {
	if (y._syncData.switching) {
		std::vector<t_index>::iterator iter;
		DispVectorS *yS = (DispVectorS*) &y;
		int i = 0;
		for (iter = bc.LoadedNodesInd.begin();iter != bc.LoadedNodesInd.end();iter++) {
			yS->cpuVec[*iter] += bc.LoadedNodes[i];
			i++;
		}
	} else {
		int blockcount = bc.LoadedNodesInd.size()/THREADS_PER_BLOCK + 1;
		AddBoundariesKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(bc.LoadedNodesInd.size(),bc.devLoadedNodesInd,bc.devLoadedNodes,y.GetVector());
	}
}

__global__ void AddPartVectorAtomicKernel(t_index numStarts,t_double *a,const t_double *__restrict__ b) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	AtomicAdd2(&a[index],b[index]);
}

#ifdef REORDER
void AddPartVectorAtomicK(t_index numStarts,t_double *dev_a,t_double *dev_b,cudaStream_t stream) {
	int blockcount = numStarts/THREADS_PER_BLOCK + 1;
	AddPartVectorAtomicKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(numStarts,dev_a,dev_b);
}

#else

__global__ void GatherSendBufferKernel(t_index numStarts,t_index *inds,t_double *x,t_double *buff) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	buff[index*3] = x[inds[index]*3];
	buff[index*3 + 1] = x[inds[index]*3 + 1];
	buff[index*3 + 2] = x[inds[index]*3 + 2];
}

void GatherSendBufferK(t_index numStarts,t_index *dev_inds,t_double *dev_x,t_double *dev_buff) {
	int blockcount = numStarts/THREADS_PER_BLOCK + 1;
	GatherSendBufferKernel<<<blockcount,THREADS_PER_BLOCK>>>(numStarts,dev_inds,dev_x,dev_buff);
}

__global__ void AddInBufferKernel(t_index numStarts,t_index *inds,t_double *x,t_double *buff) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	AtomicAdd2(&x[inds[index]*3],buff[index*3]);
	AtomicAdd2(&x[inds[index]*3 + 1],buff[index*3 + 1]);
	AtomicAdd2(&x[inds[index]*3 + 2],buff[index*3 + 2]);
}

void AddInBufferK(t_index numStarts,t_index *dev_inds,t_double *dev_x,t_double *dev_buff) {
	int blockcount = numStarts/THREADS_PER_BLOCK + 1;
	AddInBufferKernel<<<blockcount,THREADS_PER_BLOCK>>>(numStarts,dev_inds,dev_x,dev_buff);
}

__global__ void AddInBufferCompKernel(t_index numStarts,t_index *inds,t_double *x,t_double *buff) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	AtomicAdd2(&x[inds[index]],buff[index*3]);
}

void AddInBufferCompK(t_index numStarts,t_index *dev_inds,t_double *dev_x,t_double *dev_buff) {
	int blockcount = numStarts/THREADS_PER_BLOCK + 1;
	AddInBufferCompKernel<<<blockcount,THREADS_PER_BLOCK>>>(numStarts,dev_inds,dev_x,dev_buff);
}

#endif
