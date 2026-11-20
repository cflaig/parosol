/*
 * VectorKernels.cu
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
#include <omp.h>
#include "Config.h"
#include "VectorKernels.h"

__device__ t_double AtomicAdd3(t_double *address,t_double val) {
	unsigned long long int *address_as_ull = (unsigned long long int*) address;
	unsigned long long int old = *address_as_ull;
	unsigned long long int assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull,assumed,__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__global__ void AddVectorKernel(t_index numStarts,t_double *a,const t_double *__restrict__ b) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	a[index] = a[index] + b[index];
}

//a = a + b
void tAddVectorK(DispVector &a,DispVector &b,cudaStream_t stream) {
	if (a._syncData.switching) {
		DispVectorS *aS = (DispVectorS*) &a;
		DispVectorS *bS = (DispVectorS*) &b;
#pragma omp parallel for
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			aS->cpuVec[i] += bS->cpuVec[i];
		}
	} else {
		int blockcount = a._syncData.gpuVecSize/THREADS_PER_BLOCK + 1;
		AddVectorKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(a._syncData.gpuVecSize,a.GetVector(),b.GetVector());
	}
}

__global__ void SubtractVectorKernel(t_index numStarts,t_double *a,const t_double *__restrict__ b) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	a[index] = a[index] - b[index];
}

//a = a - b
void tSubtractVectorK(DispVector &a,DispVector &b,cudaStream_t stream) {
	if (a._syncData.switching) {
		DispVectorS *aS = (DispVectorS*) &a;
		DispVectorS *bS = (DispVectorS*) &b;
#pragma omp parallel for
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			aS->cpuVec[i] -= bS->cpuVec[i];
		}
	} else {
		int blockcount = a._syncData.gpuVecSize/THREADS_PER_BLOCK + 1;
		SubtractVectorKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(a._syncData.gpuVecSize,a.GetVector(),b.GetVector());
	}
}

__global__ void SubtractVector2Kernel(t_index numStarts,const t_double *__restrict__ a,t_double *b) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	b[index] = a[index] - b[index];
}

//b = a - b
void tSubtractVector2K(DispVector &a,DispVector &b,cudaStream_t stream) {
	if (a._syncData.switching) {
		DispVectorS *aS = (DispVectorS*) &a;
		DispVectorS *bS = (DispVectorS*) &b;
#pragma omp parallel for
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			bS->cpuVec[i] = aS->cpuVec[i] - bS->cpuVec[i];
		}
	} else {
		int blockcount = a._syncData.gpuVecSize/THREADS_PER_BLOCK + 1;
		SubtractVector2Kernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(a._syncData.gpuVecSize,a.GetVector(),b.GetVector());
	}
}

__global__ void ScaleVectorKernel(t_index numStarts,t_double alpha,t_double *a) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	a[index] *= alpha;
}

//a = alpha*a
void tScaleVectorK(t_double alpha,DispVector &a,cudaStream_t stream) {
	if (a._syncData.switching) {
		DispVectorS *aS = (DispVectorS*) &a;
#pragma omp parallel for
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			aS->cpuVec[i] *= alpha;
		}
	} else {
		int blockcount = a._syncData.gpuVecSize/THREADS_PER_BLOCK + 1;
		ScaleVectorKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(a._syncData.gpuVecSize,alpha,a.GetVector());
	}
}

__global__ void AxpyKernel(t_index numStarts,t_double a,const t_double *__restrict__ x,t_double *y) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	y[index] += a*x[index];
}

//y = a*x + y
void tAxpyK(t_double a,DispVector &x,DispVector &y,cudaStream_t stream) {
	if (x._syncData.switching) {
		DispVectorS *xS = (DispVectorS*) &x;
		DispVectorS *yS = (DispVectorS*) &y;
#pragma omp parallel for
		for (uint i = 0;i < xS->_cpuVecSize;i++) {
			yS->cpuVec[i] += a*xS->cpuVec[i];
		}
	} else {
		int blockcount = x._syncData.gpuVecSize/THREADS_PER_BLOCK + 1;
		AxpyKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(x._syncData.gpuVecSize,a,x.GetVector(),y.GetVector());
	}
}

__global__ void Axpy2Kernel(t_index numStarts,t_double a,t_double *x,const t_double *__restrict__ y) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	x[index] = a*x[index] + y[index];
}

//x = a*x + y
void tAxpy2K(t_double a,DispVector &x,DispVector &y,cudaStream_t stream) {
	if (x._syncData.switching) {
		DispVectorS *xS = (DispVectorS*) &x;
		DispVectorS *yS = (DispVectorS*) &y;
#pragma omp parallel for
		for (uint i = 0;i < xS->_cpuVecSize;i++) {
			xS->cpuVec[i] = a*xS->cpuVec[i] + yS->cpuVec[i];
		}
	} else {
		int blockcount = x._syncData.gpuVecSize/THREADS_PER_BLOCK + 1;
		Axpy2Kernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(x._syncData.gpuVecSize,a,x.GetVector(),y.GetVector());
	}
}

__global__ void CWiseProductVectorKernel(t_index numStarts,const t_double *__restrict__ a,const t_double *__restrict__ b,t_double *c) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	c[index] = a[index]*b[index];
}

__global__ void CWiseProductVectorCompKernel(t_index numStarts,const t_double *__restrict__ a,const t_double *__restrict__ b,t_double *c) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	c[index] = a[index/3]*b[index];
}

__global__ void CWiseProductVectorComp2Kernel(t_index numStarts,const t_index *__restrict__ fixedInds,const t_double *__restrict__ b,t_double *c) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	c[fixedInds[index]] = b[fixedInds[index]];
}

//c = a*b (componentwise)
void tCWiseProductVectorK(Diagonal &a,DispVector &b,DispVector &c,cudaStream_t stream) {
	if (a._syncData.switching) {
		DiagonalS *aS = (DiagonalS*) &a;
		DispVectorS *bS = (DispVectorS*) &b;
		DispVectorS *cS = (DispVectorS*) &c;
#pragma omp parallel for
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			cS->cpuVec[i] = aS->cpuVec[i]*bS->cpuVec[i];
		}
	} else {
		if (a._syncData.compressDia) {
			DiagonalNComp *aN = (DiagonalNComp*) &a;
			int blockcount = a._syncData.gpuVecSize/THREADS_PER_BLOCK + 1;
			CWiseProductVectorCompKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(a._syncData.gpuVecSize,a.GetVector(),b.GetVector(),c.GetVector());
			blockcount = aN->_bc->FixedNodesInd.size()/THREADS_PER_BLOCK + 1;
			CWiseProductVectorComp2Kernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(aN->_bc->FixedNodesInd.size(),aN->_bc->devFixedNodesInd,b.GetVector(),c.GetVector());
		} else {
			int blockcount = a._syncData.gpuVecSize/THREADS_PER_BLOCK + 1;
			CWiseProductVectorKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(a._syncData.gpuVecSize,a.GetVector(),b.GetVector(),c.GetVector());
		}
	}
}

__global__ void CWiseProductSVectorKernel(t_index numStarts,t_double alpha,const t_double *__restrict__ a,const t_double *__restrict__ b,t_double *c) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	c[index] = a[index]*b[index]*alpha;
}

__global__ void CWiseProductSVectorCompKernel(t_index numStarts,t_double alpha,const t_double *__restrict__ a,const t_double *__restrict__ b,t_double *c) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	c[index] = a[index/3]*b[index]*alpha;
}

__global__ void CWiseProductSVectorComp2Kernel(t_index numStarts,t_double alpha,const t_index *__restrict__ fixedInds,const t_double *__restrict__ b,t_double *c) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	c[fixedInds[index]] = b[fixedInds[index]]*alpha;
}

//c = a*b*alpha (componentwise)
void tCWiseProductSVectorK(t_double alpha,Diagonal &a,DispVector &b,DispVector &c,cudaStream_t stream) {
	if (a._syncData.switching) {
		DiagonalS *aS = (DiagonalS*) &a;
		DispVectorS *bS = (DispVectorS*) &b;
		DispVectorS *cS = (DispVectorS*) &c;
#pragma omp parallel for
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			cS->cpuVec[i] = aS->cpuVec[i]*bS->cpuVec[i]*alpha;
		}
	} else {
		if (a._syncData.compressDia) {
			DiagonalNComp *aN = (DiagonalNComp*) &a;
			int blockcount = a._syncData.gpuVecSize/THREADS_PER_BLOCK + 1;
			CWiseProductSVectorCompKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(a._syncData.gpuVecSize,alpha,a.GetVector(),b.GetVector(),c.GetVector());
			blockcount = aN->_bc->FixedNodesInd.size()/THREADS_PER_BLOCK + 1;
			CWiseProductSVectorComp2Kernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(aN->_bc->FixedNodesInd.size(),alpha,aN->_bc->devFixedNodesInd,b.GetVector(),c.GetVector());
		} else {
			int blockcount = a._syncData.gpuVecSize/THREADS_PER_BLOCK + 1;
			CWiseProductSVectorKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(a._syncData.gpuVecSize,alpha,a.GetVector(),b.GetVector(),c.GetVector());
		}
	}
}

__global__ void CWiseProductS2VectorKernel(t_index numStarts,t_double alpha,const t_double *__restrict__ a,const t_double *__restrict__ b,const t_double *__restrict__ c,t_double *d,t_double *e) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	d[index] = a[index]*(b[index] - c[index])*alpha;
	e[index] += d[index];
}

__global__ void CWiseProductS2VectorCompKernel(t_index numStarts,t_double alpha,const t_double *__restrict__ a,const t_double *__restrict__ b,const t_double *__restrict__ c,t_double *d) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	d[index] = a[index/3]*(b[index] - c[index])*alpha;
}

__global__ void CWiseProductS2VectorComp2Kernel(t_index numStarts,t_double alpha,const t_index *__restrict__ fixedInds,const t_double *__restrict__ b,const t_double *__restrict__ c,t_double *d) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	d[fixedInds[index]] = (b[fixedInds[index]] - c[fixedInds[index]])*alpha;
}

//d = a*(b-c)*alpha, e = e + d (componentwise)
void tCWiseProductS2VectorK(t_double alpha,Diagonal &a,DispVector &b,DispVector &c,DispVector &d,DispVector &e,cudaStream_t stream) {
	if (a._syncData.switching) {
		DiagonalS *aS = (DiagonalS*) &a;
		DispVectorS *bS = (DispVectorS*) &b;
		DispVectorS *cS = (DispVectorS*) &c;
		DispVectorS *dS = (DispVectorS*) &d;
		DispVectorS *eS = (DispVectorS*) &e;
#pragma omp parallel for
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			dS->cpuVec[i] = aS->cpuVec[i]*(bS->cpuVec[i] - cS->cpuVec[i])*alpha;
			eS->cpuVec[i] += dS->cpuVec[i];
		}
	} else {
		if (a._syncData.compressDia) {
			DiagonalNComp *aN = (DiagonalNComp*) &a;
			int blockcount = a._syncData.gpuVecSize/THREADS_PER_BLOCK + 1;
			CWiseProductS2VectorCompKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(a._syncData.gpuVecSize,alpha,a.GetVector(),b.GetVector(),c.GetVector(),d.GetVector());
			blockcount = aN->_bc->FixedNodesInd.size()/THREADS_PER_BLOCK + 1;
			CWiseProductS2VectorComp2Kernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(aN->_bc->FixedNodesInd.size(),alpha,aN->_bc->devFixedNodesInd,b.GetVector(),c.GetVector(),d.GetVector());
			blockcount = b._syncData.gpuVecSize/THREADS_PER_BLOCK + 1;
			AddVectorKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(b._syncData.gpuVecSize,e.GetVector(),d.GetVector());
		} else {
			int blockcount = a._syncData.gpuVecSize/THREADS_PER_BLOCK + 1;
			CWiseProductS2VectorKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(a._syncData.gpuVecSize,alpha,a.GetVector(),b.GetVector(),c.GetVector(),d.GetVector(),e.GetVector());
		}
	}
}

__global__ void CWiseProductS3VectorKernel(t_index numStarts,t_double alpha,const t_double *__restrict__ a,const t_double *__restrict__ b,const t_double *__restrict__ c,t_double *d) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	d[index] += a[index]*(b[index] - c[index])*alpha;
}

__global__ void CWiseProductS3VectorCompKernel(t_index numStarts,t_double alpha,const t_double *__restrict__ a,const t_double *__restrict__ b,const t_double *__restrict__ c,t_double *d) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	d[index] += a[index/3]*(b[index] - c[index])*alpha;
}

__global__ void CWiseProductS3VectorComp2Kernel(t_index numStarts,t_double alpha,const t_index *__restrict__ fixedInds,const t_double *__restrict__ a,const t_double *__restrict__ b,const t_double *__restrict__ c,t_double *d) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	d[fixedInds[index]] -= a[fixedInds[index]/3]*(b[fixedInds[index]] - c[fixedInds[index]])*alpha;
	d[fixedInds[index]] += (b[fixedInds[index]] - c[fixedInds[index]])*alpha;
}

//d = a*(b - c)*alpha + d (componentwise)
void tCWiseProductS3VectorK(t_double alpha,Diagonal &a,DispVector &b,DispVector &c,DispVector &d,cudaStream_t stream) {
	if (a._syncData.switching) {
		DiagonalS *aS = (DiagonalS*) &a;
		DispVectorS *bS = (DispVectorS*) &b;
		DispVectorS *cS = (DispVectorS*) &c;
		DispVectorS *dS = (DispVectorS*) &d;
#pragma omp parallel for
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			dS->cpuVec[i] += aS->cpuVec[i]*(bS->cpuVec[i] - cS->cpuVec[i])*alpha;
		}
	} else {
		if (a._syncData.compressDia) {
			DiagonalNComp *aN = (DiagonalNComp*) &a;
			int blockcount = a._syncData.gpuVecSize/THREADS_PER_BLOCK + 1;
			CWiseProductS3VectorCompKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(a._syncData.gpuVecSize,alpha,a.GetVector(),b.GetVector(),c.GetVector(),d.GetVector());
			blockcount = aN->_bc->FixedNodesInd.size()/THREADS_PER_BLOCK + 1;
			CWiseProductS3VectorComp2Kernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(aN->_bc->FixedNodesInd.size(),alpha,aN->_bc->devFixedNodesInd,a.GetVector(),b.GetVector(),c.GetVector(),d.GetVector());
		} else {
			int blockcount = a._syncData.gpuVecSize/THREADS_PER_BLOCK + 1;
			CWiseProductS3VectorKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(a._syncData.gpuVecSize,alpha,a.GetVector(),b.GetVector(),c.GetVector(),d.GetVector());
		}
	}
}

__global__ void CWiseProductS4VectorKernel(t_index numStarts,t_double alpha,t_double beta,const t_double *__restrict__ a,const t_double *__restrict__ b,const t_double *__restrict__ c,t_double *d,t_double *e) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	d[index] = d[index]*beta + a[index]*(b[index] - c[index])*alpha;
	e[index] += d[index];
}

__global__ void CWiseProductS4VectorCompKernel(t_index numStarts,t_double alpha,t_double beta,const t_double *__restrict__ a,const t_double *__restrict__ b,const t_double *__restrict__ c,t_double *d) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	d[index] = d[index]*beta + a[index/3]*(b[index] - c[index])*alpha;
}

__global__ void CWiseProductS4VectorComp2Kernel(t_index numStarts,t_double alpha,const t_index *__restrict__ fixedInds,const t_double *__restrict__ a,const t_double *__restrict__ b,const t_double *__restrict__ c,t_double *d) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	d[fixedInds[index]] -= a[fixedInds[index]/3]*(b[fixedInds[index]] - c[fixedInds[index]])*alpha;
	d[fixedInds[index]] += (b[fixedInds[index]] - c[fixedInds[index]])*alpha;
}

//d = a*(b - c)*alpha + d*beta, e = e + d (componentwise)
void tCWiseProductS4VectorK(t_double alpha,t_double beta,Diagonal &a,DispVector &b,DispVector &c,DispVector &d,DispVector &e,cudaStream_t stream) {
	if (a._syncData.switching) {
		DiagonalS *aS = (DiagonalS*) &a;
		DispVectorS *bS = (DispVectorS*) &b;
		DispVectorS *cS = (DispVectorS*) &c;
		DispVectorS *dS = (DispVectorS*) &d;
		DispVectorS *eS = (DispVectorS*) &e;
#pragma omp parallel for
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			dS->cpuVec[i] = aS->cpuVec[i]*(bS->cpuVec[i] - cS->cpuVec[i])*alpha + dS->cpuVec[i]*beta;
			eS->cpuVec[i] += dS->cpuVec[i];
		}
	} else {
		if (a._syncData.compressDia) {
			DiagonalNComp *aN = (DiagonalNComp*) &a;
			int blockcount = a._syncData.gpuVecSize/THREADS_PER_BLOCK + 1;
			CWiseProductS4VectorCompKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(a._syncData.gpuVecSize,alpha,beta,a.GetVector(),b.GetVector(),c.GetVector(),d.GetVector());
			blockcount = aN->_bc->FixedNodesInd.size()/THREADS_PER_BLOCK + 1;
			CWiseProductS4VectorComp2Kernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(aN->_bc->FixedNodesInd.size(),alpha,aN->_bc->devFixedNodesInd,a.GetVector(),b.GetVector(),c.GetVector(),d.GetVector());
			blockcount = d._syncData.gpuVecSize/THREADS_PER_BLOCK + 1;
			AddVectorKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(d._syncData.gpuVecSize,e.GetVector(),d.GetVector());
		} else {
			int blockcount = a._syncData.gpuVecSize/THREADS_PER_BLOCK + 1;
			CWiseProductS4VectorKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(a._syncData.gpuVecSize,alpha,beta,a.GetVector(),b.GetVector(),c.GetVector(),d.GetVector(),e.GetVector());
		}
	}
}

__global__ void DotKernel(t_index numStarts,const t_double *__restrict__ a,const t_double *__restrict__ b,t_double *c) {
	extern __shared__ t_double tmp[];
	tmp[threadIdx.x] = 0;
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < numStarts) {
		tmp[threadIdx.x] = a[index] * b[index];
	}
	__syncthreads();

	if (threadIdx.x == 0 && index < numStarts) {
		t_double res = 0;
		for (int i = 0;i < blockDim.x;i++) {
			res += tmp[i];
		}
		c[blockIdx.x] = res;
	}
}

t_double tDotK(DispVector &a,DispVector &b,t_double *c,t_double *dev_c,t_index blockSize,cudaStream_t stream) {
	t_double res = 0;
	if (a._syncData.switching) {
		DispVectorS *aS = (DispVectorS*) &a;
		DispVectorS *bS = (DispVectorS*) &b;
#pragma omp parallel for reduction(+:res)
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			res += aS->cpuVec[i]*bS->cpuVec[i];
		}
	} else {
		int blockcount = (int) ceil((double) a._syncData.dotStartSize/blockSize);
		DotKernel<<<blockcount,blockSize,blockSize*sizeof(t_double),stream>>>(a._syncData.dotStartSize,a.GetVector(),b.GetVector(),dev_c);
		CCS(cudaMemcpyAsync(c,dev_c,blockcount*sizeof(t_double),cudaMemcpyDeviceToHost,stream),"DotK 1");
		CCS(cudaStreamSynchronize(stream),"DotK 2");
		for (int i = 0;i < blockcount;i++) {
			res += c[i];
		}
	}
	return res;
}

__global__ void Dot2Kernel(t_index numStarts,const t_double *__restrict__ a,t_double *c) {
	extern __shared__ t_double tmp[];
	tmp[threadIdx.x] = 0;
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < numStarts) {
		tmp[threadIdx.x] = a[index];
		tmp[threadIdx.x] *= tmp[threadIdx.x];
	}
	__syncthreads();

	if (threadIdx.x == 0 && index < numStarts) {
		t_double res = 0;
		for (int i = 0;i < blockDim.x;i++) {
			res += tmp[i];
		}
		c[blockIdx.x] = res;
	}
}

t_double tDot2K(DispVector &a,t_double *c,t_double *dev_c,t_index blockSize,cudaStream_t stream) {
	t_double res = 0;
	if (a._syncData.switching) {
		DispVectorS *aS = (DispVectorS*) &a;
#pragma omp parallel for reduction(+:res)
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			res += aS->cpuVec[i]*aS->cpuVec[i];
		}
	} else {
		int blockcount = (int) ceil((double) a._syncData.dotStartSize/blockSize);
		Dot2Kernel<<<blockcount,blockSize,blockSize*sizeof(t_double),stream>>>(a._syncData.dotStartSize,a.GetVector(),dev_c);
		CCS(cudaMemcpyAsync(c,dev_c,blockcount*sizeof(t_double),cudaMemcpyDeviceToHost,stream),"Dot2K 1")
		CCS(cudaStreamSynchronize(stream),"Dot2K 2");
		for (int i = 0;i < blockcount;i++) {
			res += c[i];
		}
	}
	return res;
}

__global__ void PInverseDiaKernel(t_index numStarts,uint *hasZeros,t_double *dia) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	t_double d = dia[index];
	if (d != 0) {
		dia[index] = 1.0/d;
	} else {
		atomicInc(hasZeros,1);
	}
}

bool tPInverseDiaK(Diagonal &dia,cudaStream_t stream) {
	bool hasZeros = false;
	if (dia._syncData.switching) {
		DiagonalS *diaS = (DiagonalS*) &dia;
#pragma omp parallel for
		for (uint i = 0;i < diaS->_cpuVecSize;i++) {
			if (diaS->cpuVec[i] != 0) {
				diaS->cpuVec[i] = 1.0/diaS->cpuVec[i];
			} else {
				hasZeros = true;
			}
		}
	} else {
		int blockcount;
		uint *hZeros;
		CCS(cudaMalloc((void**) &hZeros,sizeof(uint)),"PInverseDiaK 1");
		CCS(cudaMemset(hZeros,0,sizeof(uint)),"PInverseDiaK 2");
		if (dia._syncData.compressDia) {
			DiagonalNComp *diaN = (DiagonalNComp*) &dia;
			blockcount = diaN->_gpuVecSize/THREADS_PER_BLOCK + 1;
			PInverseDiaKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(diaN->_gpuVecSize,hZeros,dia.GetVector());
		} else {
			blockcount = dia._syncData.gpuVecSize/THREADS_PER_BLOCK + 1;
			PInverseDiaKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(dia._syncData.gpuVecSize,hZeros,dia.GetVector());
		}
		int hZ = 0;
		CCS(cudaMemcpy(&hZ,hZeros,sizeof(uint),cudaMemcpyDeviceToHost),"PInverseDiaK 3");
		CCS(cudaFree(hZeros),"PInverseDiaK 4");
		if (hZ) {
			hasZeros = true;
		}
	}
	return hasZeros;
}
