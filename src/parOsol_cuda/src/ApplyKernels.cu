/*
 * ApplyKernels.cu
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
#include <stdio.h>
#include <vector>
#include "Config.h"
#include "ApplyKernels.h"

__device__ __constant__ t_double sMat[12*576];
__device__ __constant__ Off8 rpLookupTable[8];

void SetLocalStiffnessMatrix(t_double *smat,int offset) {
	CCS(cudaMemcpyToSymbol(sMat,smat,24*24*sizeof(t_double),offset*sizeof(t_double),cudaMemcpyHostToDevice),"SetLocalStiffnessMatrix 1");
}

void SetRpLookupTable() {
	std::vector<Off8> table;
	table.push_back(Off8(1,0,0,0));
	table.push_back(Off8(2,1,0,0));
	table.push_back(Off8(2,3,0,0));
	table.push_back(Off8(4,1,2,3));
	table.push_back(Off8(2,4,0,0));
	table.push_back(Off8(4,1,4,5));
	table.push_back(Off8(4,3,4,7));
	table.push_back(Off8(8,0,0,0));
	CCS(cudaMemcpyToSymbol(rpLookupTable,table.data(),8*sizeof(Off8),0,cudaMemcpyHostToDevice),"SetRpLookupTable 1");

}

__device__ t_double AtomicAdd(t_double *address,t_double val) {
	unsigned long long int *address_as_ull = (unsigned long long int*) address;
	unsigned long long int old = *address_as_ull;
	unsigned long long int assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull,assumed,__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__device__ t_index GetOffsetOfKey(t_octree_key key,HashMask hashMask) {
	offsetD *tKey = (offsetD*) &key;

	//not sure if it works correct
	int hashIndex = tKey->asuInt[0] & (NUMHASH - 1);

	t_index off = (tKey->asuInt[0] % hashMask.dev_hashMaskGpu[hashIndex].offsetTableSize) + hashMask.dev_hashMaskGpu[hashIndex].offsetTableMod*(tKey->asuInt[1] % hashMask.dev_hashMaskGpu[hashIndex].offsetTableSize);
	off = off % hashMask.dev_hashMaskGpu[hashIndex].offsetTableSize;
	t_offset offset = hashMask.dev_hashMaskGpu[hashIndex].dev_offsetTable[off];

	t_octree_key sKey = key + offset;
	tKey = (offsetD*) &sKey;
	off = (tKey->asuInt[0] % hashMask.dev_hashMaskGpu[hashIndex].hashTableSize) + hashMask.dev_hashMaskGpu[hashIndex].hashTableMod*(tKey->asuInt[1] % hashMask.dev_hashMaskGpu[hashIndex].hashTableSize);
	off = off % hashMask.dev_hashMaskGpu[hashIndex].hashTableSize;
	return hashMask.dev_hashMaskGpu[hashIndex].dev_hashTable[off]*3;
}

__device__ void GetKeys(t_octree_key key,HashMask hashMask,t_octree_key ks[8]) {
	ks[0] = key;

	t_octree_key tmp = (ks[0] | hashMask.maskx) + 1;
	t_octree_key diffx = (tmp&~hashMask.maskx) | (ks[0]&hashMask.maskx);
	diffx = diffx ^ ks[0];
	tmp = (ks[0] | hashMask.masky) + 1;
	t_octree_key diffy = (tmp&~hashMask.masky) | (ks[0]&hashMask.masky);
	diffy = diffy ^ ks[0];
	tmp = (ks[0] | hashMask.maskz) + 1;
	t_octree_key diffz = (tmp&~hashMask.maskz) | (ks[0]&hashMask.maskz);
	diffz = diffz ^ ks[0];

	ks[1] = ks[0]^diffx;
	ks[2] = ks[1]^diffy;
	ks[3] = ks[0]^diffy;
	ks[4] = ks[0]^diffz;
	ks[5] = ks[1]^diffz;
	ks[6] = ks[2]^diffz;
	ks[7] = ks[3]^diffz;
}

__device__ void GetOffsetsFromVector(t_index index,ElementOffsets elementOffsets,t_index offsets[8]) {
#ifdef COMPUTE35
	if (index < elementOffsets.breakPoint0) {
		offsets[0] = __ldg(&elementOffsets.dev_startInds[index])*3;
		offsetD l = elementOffsets.dev_offsets_long1[index];
		offsets[1] = l.asuInt[0]*3;
		offsets[2] = l.asuInt[1]*3;
		l = elementOffsets.dev_offsets_long2[index];
		offsets[3] = l.asuInt[0]*3;
		offsets[4] = l.asuInt[1]*3;
		l = elementOffsets.dev_offsets_long3[index];
		offsets[5] = l.asuInt[0]*3;
		offsets[6] = l.asuInt[1]*3;
		offsets[7] = __ldg(&elementOffsets.dev_offsets_last[index])*3;
	} else if (index < elementOffsets.breakPoint1) {
		offsets[0] = __ldg(&elementOffsets.dev_startInds[index])*3;
		offsetD l = elementOffsets.dev_offsets_long1[index];
		for (int i = 1;i < 4;i++) {
			offsets[i] = offsets[0] + l.asuShort[i]*3;
		}
		l = elementOffsets.dev_offsets_long2[index];
		for (int i = 0;i < 4;i++) {
			offsets[i + 4] = offsets[0] + l.asuShort[i]*3;
		}
	} else {
		offsets[0] = __ldg(&elementOffsets.dev_startInds[index])*3;
		offsetD l = elementOffsets.dev_offsets_long1[index];
		for (int i = 1;i < 8;i++) {
			offsets[i] = offsets[0] + l.asuChar[i]*3;
		}
	}
#else
	if (index < elementOffsets.breakPoint0) {
		offsets[0] = elementOffsets.dev_startInds[index]*3;
		offsetD l = elementOffsets.dev_offsets_long1[index];
		offsets[1] = l.asuInt[0]*3;
		offsets[2] = l.asuInt[1]*3;
		l = elementOffsets.dev_offsets_long2[index];
		offsets[3] = l.asuInt[0]*3;
		offsets[4] = l.asuInt[1]*3;
		l = elementOffsets.dev_offsets_long3[index];
		offsets[5] = l.asuInt[0]*3;
		offsets[6] = l.asuInt[1]*3;
		offsets[7] = elementOffsets.dev_offsets_last[index]*3;
	} else if (index < elementOffsets.breakPoint1) {
		offsets[0] = elementOffsets.dev_startInds[index]*3;
		offsetD l = elementOffsets.dev_offsets_long1[index];
		for (int i = 1;i < 4;i++) {
			offsets[i] = offsets[0] + l.asuShort[i]*3;
		}
		l = elementOffsets.dev_offsets_long2[index];
		for (int i = 0;i < 4;i++) {
			offsets[i + 4] = offsets[0] + l.asuShort[i]*3;
		}
	} else {
		offsets[0] = elementOffsets.dev_startInds[index]*3;
		offsetD l = elementOffsets.dev_offsets_long1[index];
		for (int i = 1;i < 8;i++) {
			offsets[i] = offsets[0] + l.asuChar[i]*3;
		}
	}
#endif
}

__device__ uchar GetOffsetsFromVectorRP(t_index index,ElementOffsets elementOffsets,t_index offsets[8]) {
	uchar localInds;
#ifdef COMPUTE35
	if (index < elementOffsets.breakPoint0) {
		offsets[0] = __ldg(&elementOffsets.dev_startInds[index])*3;
		offsetD l = elementOffsets.dev_offsets_long1[index];
		offsets[1] = l.asuInt[0]*3;
		offsets[2] = l.asuInt[1]*3;
		l = elementOffsets.dev_offsets_long2[index];
		offsets[3] = l.asuInt[0]*3;
		offsets[4] = l.asuInt[1]*3;
		l = elementOffsets.dev_offsets_long3[index];
		offsets[5] = l.asuInt[0]*3;
		offsets[6] = l.asuInt[1]*3;
		offsets[7] = __ldg(&elementOffsets.dev_offsets_last[index])*3;
		localInds = __ldg(&elementOffsets.dev_localInds[index]);
	} else if (index < elementOffsets.breakPoint1) {
		offsets[0] = __ldg(&elementOffsets.dev_startInds[index])*3;
		offsetD o = elementOffsets.dev_offsets_long1[index];
		localInds = (uchar) o.asuShort[0];
		for (int i = 1;i < 4;i++) {
			offsets[i] = offsets[0] + o.asuShort[i]*3;
		}
		o = elementOffsets.dev_offsets_long2[index];
		for (int i = 0;i < 4;i++) {
			offsets[i + 4] = offsets[0] + o.asuShort[i]*3;
		}
	} else if (index < elementOffsets.breakPoint2) {
		offsets[0] = __ldg(&elementOffsets.dev_startInds[index])*3;
		offsetD o = elementOffsets.dev_offsets_long1[index];
		localInds = o.asuChar[0];
		for (int i = 1;i < 8;i++) {
			offsets[i] = offsets[0] + o.asuChar[i]*3;
		}
	} else {
		int offset1 = index - (elementOffsets.breakPoint2 - elementOffsets.breakPoint1);
		int offset2 = index - (elementOffsets.breakPoint2 - elementOffsets.breakPoint0);
		offsets[0] = __ldg(&elementOffsets.dev_startInds[index])*3;
		offsetD l = elementOffsets.dev_offsets_long1[index];
		offsets[1] = l.asuInt[0]*3;
		offsets[2] = l.asuInt[1]*3;
		l = elementOffsets.dev_offsets_long2[offset1];
		offsets[3] = l.asuInt[0]*3;
		offsets[4] = l.asuInt[1]*3;
		l = elementOffsets.dev_offsets_long3[offset2];
		offsets[5] = l.asuInt[0]*3;
		offsets[6] = l.asuInt[1]*3;
		offsets[7] = __ldg(&elementOffsets.dev_offsets_last[offset2])*3;
		localInds = __ldg(&elementOffsets.dev_localInds[offset2]);
	}
#else
	if (index < elementOffsets.breakPoint0) {
		offsets[0] = elementOffsets.dev_startInds[index]*3;
		offsetD l = elementOffsets.dev_offsets_long1[index];
		offsets[1] = l.asuInt[0]*3;
		offsets[2] = l.asuInt[1]*3;
		l = elementOffsets.dev_offsets_long2[index];
		offsets[3] = l.asuInt[0]*3;
		offsets[4] = l.asuInt[1]*3;
		l = elementOffsets.dev_offsets_long3[index];
		offsets[5] = l.asuInt[0]*3;
		offsets[6] = l.asuInt[1]*3;
		offsets[7] = elementOffsets.dev_offsets_last[index]*3;
		localInds = elementOffsets.dev_localInds[index];
	} else if (index < elementOffsets.breakPoint1) {
		offsets[0] = elementOffsets.dev_startInds[index]*3;
		offsetD o = elementOffsets.dev_offsets_long1[index];
		localInds = (uchar) o.asuShort[0];
		for (int i = 1;i < 4;i++) {
			offsets[i] = offsets[0] + o.asuShort[i]*3;
		}
		o = elementOffsets.dev_offsets_long2[index];
		for (int i = 0;i < 4;i++) {
			offsets[i + 4] = offsets[0] + o.asuShort[i]*3;
		}
	} else if (index < elementOffsets.breakPoint2) {
		offsets[0] = elementOffsets.dev_startInds[index]*3;
		offsetD o = elementOffsets.dev_offsets_long1[index];
		localInds = o.asuChar[0];
		for (int i = 1;i < 8;i++) {
			offsets[i] = offsets[0] + o.asuChar[i]*3;
		}
	} else {
		int offset1 = index - (elementOffsets.breakPoint2 - elementOffsets.breakPoint1);
		int offset2 = index - (elementOffsets.breakPoint2 - elementOffsets.breakPoint0);
		offsets[0] = elementOffsets.dev_startInds[index]*3;
		offsetD l = elementOffsets.dev_offsets_long1[index];
		offsets[1] = l.asuInt[0]*3;
		offsets[2] = l.asuInt[1]*3;
		l = elementOffsets.dev_offsets_long2[offset1];
		offsets[3] = l.asuInt[0]*3;
		offsets[4] = l.asuInt[1]*3;
		l = elementOffsets.dev_offsets_long3[offset2];
		offsets[5] = l.asuInt[0]*3;
		offsets[6] = l.asuInt[1]*3;
		offsets[7] = elementOffsets.dev_offsets_last[offset2]*3;
		localInds = elementOffsets.dev_localInds[offset2];
	}
#endif
	return localInds;
}

__device__ void MatMul(t_index *offsets,t_double weight,t_double *disps,t_double *smat,t_double *y) {
	for (int i = 0;i < 8;i++) {
		for (int j = 0;j < 3;j++) {
			t_double res = 0;
			int of = (i*3 + j)*24;
			for (int k = 0;k < 24;k++) {
				res += smat[of + k]*disps[k];
			}
			AtomicAdd(&y[offsets[i] + j],res*weight);
		}
	}
}

__device__ void MatMulC(t_index *offsets,t_double weight,t_double *disps,int matOffset,t_double *y) {
	for (int i = 0;i < 8;i++) {
		for (int j = 0;j < 3;j++) {
			t_double res = 0;
			int of = (i*3 + j)*24 + matOffset;
			for (int k = 0;k < 24;k++) {
				res += sMat[of + k]*disps[k];
			}
			AtomicAdd(&y[offsets[i] + j],res*weight);
		}
	}
}

__global__ void MatrixVectorHashKernel(t_index numKeys,HashMask hashMask,const t_double *__restrict__ x,t_double *y,t_double *smat) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numKeys) {
		return;
	}
	t_octree_key ks[8];
	GetKeys(hashMask.dev_keys[index],hashMask,ks);

	t_index offsets[8];
	t_double disps[24];
	for (int i = 0;i < 8;i++) {

		offsets[i] = GetOffsetOfKey(ks[i],hashMask);
		disps[i*3] = x[offsets[i]];
		disps[i*3 + 1] = x[offsets[i] + 1];
		disps[i*3 + 2] = x[offsets[i] + 2];
	}
#ifdef COMPUTE35
	MatMul(offsets,__ldg(&hashMask.dev_weights[index]),disps,smat,y);
#else
	MatMul(offsets,hashMask.dev_weights[index],disps,smat,y);
#endif
}

void GPUGridHash::MatrixVectorK(t_double *dev_x,t_double *dev_y,t_double *dev_smat,cudaStream_t stream) {
	int blockcount = _applyStarts/THREADS_PER_BLOCK + 1;
	MatrixVectorHashKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(_applyStarts,_hashMask,dev_x,dev_y,dev_smat);
}

__global__ void MatrixVectorCHashKernel(t_index numStarts,HashMask hashMask,const t_double *__restrict__ x,t_double *y,int matOffset) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	__shared__ t_double result[384];
	__shared__ t_index inds[384];
	t_index offsets[8];
	t_double disps[24];
	t_double weight;

	if (index >= numStarts) {
		inds[threadIdx.x*3] = 0xffffffff;
		inds[threadIdx.x*3 + 1] = 0xffffffff;
		inds[threadIdx.x*3 + 2] = 0xffffffff;
	} else {
		t_octree_key ks[8];
		GetKeys(hashMask.dev_keys[index],hashMask,ks);
		weight = hashMask.dev_weights[index];

		for (int i = 0;i < 8;i++) {
			offsets[i] = GetOffsetOfKey(ks[i],hashMask);
		}

	}

	for (int i = 0;i < 8;i++) {
		if (index < numStarts) {
			inds[threadIdx.x*3] = offsets[i];
			inds[threadIdx.x*3 + 1] = offsets[i] + 1;
			inds[threadIdx.x*3 + 2] = offsets[i] + 2;
		}
		__syncthreads();
		if (inds[threadIdx.x] != 0xffffffff) {
			result[threadIdx.x] = x[inds[threadIdx.x]];
		}
		if (inds[threadIdx.x + blockDim.x] != 0xffffffff) {
			result[threadIdx.x + blockDim.x] = x[inds[threadIdx.x + blockDim.x]];
		}
		if (inds[threadIdx.x + 2*blockDim.x] != 0xffffffff) {
			result[threadIdx.x + 2*blockDim.x] = x[inds[threadIdx.x + 2*blockDim.x]];
		}
		__syncthreads();

		if (index < numStarts) {
			disps[i*3] = result[threadIdx.x*3];
			disps[i*3 + 1] = result[threadIdx.x*3 + 1];
			disps[i*3 + 2] = result[threadIdx.x*3 + 2];
		}
	}
	__syncthreads();

	for (int i = 0;i < 8;i++) {
		if (index < numStarts) {
			for (int j = 0;j < 3;j++) {
				t_double res = 0;
				int of = (i*3 + j)*24 + matOffset;
				for (int k = 0;k < 24;k++) {
					res += sMat[of + k]*disps[k];
				}
				result[threadIdx.x*3 + j] = res*weight;
				inds[threadIdx.x*3 + j] = offsets[i] + j;
			}
		}
		__syncthreads();
		if (inds[threadIdx.x] != 0xffffffff) {
			AtomicAdd(&y[inds[threadIdx.x]],result[threadIdx.x]);
		}
		if (inds[threadIdx.x + blockDim.x] != 0xffffffff) {
			AtomicAdd(&y[inds[threadIdx.x + blockDim.x]],result[threadIdx.x + blockDim.x]);
		}
		if (inds[threadIdx.x + 2*blockDim.x] != 0xffffffff) {
			AtomicAdd(&y[inds[threadIdx.x + 2*blockDim.x]],result[threadIdx.x + 2*blockDim.x]);
		}
		__syncthreads();
	}

}

void GPUGridHash::MatrixVectorCK(t_double *dev_x,t_double *dev_y,int matOffset,cudaStream_t stream) {
	int blockcount = _applyStarts/128 + 1;
	MatrixVectorCHashKernel<<<blockcount,128,0,stream>>>(_applyStarts,_hashMask,dev_x,dev_y,matOffset);
}

__global__ void MatrixVectorOffsetKernel(t_index numStarts,ElementOffsets elementOffsets,const t_double *__restrict__ x,t_double *y,t_double *smat) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
	t_index offsets[8];
	GetOffsetsFromVector(index,elementOffsets,offsets);

	t_double disps[24];
	for (int i = 0;i < 8;i++) {
		disps[i*3] = x[offsets[i]];
		disps[i*3 + 1] = x[offsets[i] + 1];
		disps[i*3 + 2] = x[offsets[i] + 2];
	}
#ifdef COMPUTE35
	MatMul(offsets,__ldg(&elementOffsets.dev_weights[index]),disps,smat,y);
#else
	MatMul(offsets,elementOffsets.dev_weights[index],disps,smat,y);
#endif
}

cudaError_t CheckCudaKernelImage() {
    cudaFuncAttributes attributes{};
#ifdef HASH
    return cudaFuncGetAttributes(&attributes, MatrixVectorHashKernel);
#else
    return cudaFuncGetAttributes(&attributes, MatrixVectorOffsetKernel);
#endif
}

void GPUGridOffset::MatrixVectorK(t_double *dev_x,t_double *dev_y,t_double *dev_smat,cudaStream_t stream) {
	int blockcount = _applyStarts/THREADS_PER_BLOCK + 1;
	MatrixVectorOffsetKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(_applyStarts,_elementOffsets,dev_x,dev_y,dev_smat);
}

__global__ void MatrixVectorCOffsetKernel(t_index numStarts,ElementOffsets elementOffsets,const t_double *__restrict__ x,t_double *y,int matOffset) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	__shared__ t_double result[384];
	__shared__ t_index inds[384];
	t_index offsets[8];
	t_double disps[24];
	t_double weight;

	if (index >= numStarts) {
		inds[threadIdx.x*3] = 0xffffffff;
		inds[threadIdx.x*3 + 1] = 0xffffffff;
		inds[threadIdx.x*3 + 2] = 0xffffffff;
	} else {
		GetOffsetsFromVector(index,elementOffsets,offsets);
		weight = elementOffsets.dev_weights[index];
	}

	for (int i = 0;i < 8;i++) {
		if (index < numStarts) {
			inds[threadIdx.x*3] = offsets[i];
			inds[threadIdx.x*3 + 1] = offsets[i] + 1;
			inds[threadIdx.x*3 + 2] = offsets[i] + 2;
		}
		__syncthreads();
		if (inds[threadIdx.x] != 0xffffffff) {
			result[threadIdx.x] = x[inds[threadIdx.x]];
		}
		if (inds[threadIdx.x + blockDim.x] != 0xffffffff) {
			result[threadIdx.x + blockDim.x] = x[inds[threadIdx.x + blockDim.x]];
		}
		if (inds[threadIdx.x + 2*blockDim.x] != 0xffffffff) {
			result[threadIdx.x + 2*blockDim.x] = x[inds[threadIdx.x + 2*blockDim.x]];
		}
		__syncthreads();

		if (index < numStarts) {
			disps[i*3] = result[threadIdx.x*3];
			disps[i*3 + 1] = result[threadIdx.x*3 + 1];
			disps[i*3 + 2] = result[threadIdx.x*3 + 2];
		}
	}

	__syncthreads();

	for (int i = 0;i < 8;i++) {
		if (index < numStarts) {
			for (int j = 0;j < 3;j++) {
				t_double res = 0;
				int of = (i*3 + j)*24 + matOffset;
				for (int k = 0;k < 24;k++) {
					res += sMat[of + k]*disps[k];
				}
				result[threadIdx.x*3 + j] = res*weight;
				inds[threadIdx.x*3 + j] = offsets[i] + j;
			}
		}
		__syncthreads();
		if (inds[threadIdx.x] != 0xffffffff) {
			AtomicAdd(&y[inds[threadIdx.x]],result[threadIdx.x]);
		}
		if (inds[threadIdx.x + blockDim.x] != 0xffffffff) {
			AtomicAdd(&y[inds[threadIdx.x + blockDim.x]],result[threadIdx.x + blockDim.x]);
		}
		if (inds[threadIdx.x + 2*blockDim.x] != 0xffffffff) {
			AtomicAdd(&y[inds[threadIdx.x + 2*blockDim.x]],result[threadIdx.x + 2*blockDim.x]);
		}
		__syncthreads();
	}

}

void GPUGridOffset::MatrixVectorCK(t_double *dev_x,t_double *dev_y,int matOffset,cudaStream_t stream) {
	int blockcount = _applyStarts/128 + 1;
	MatrixVectorCOffsetKernel<<<blockcount,128,0,stream>>>(_applyStarts,_elementOffsets,dev_x,dev_y,matOffset);
}

__global__ void MatrixVectorDiaHashKernel(t_index numKeys,HashMask hashMask,t_double *x,t_double smat) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numKeys) {
		return;
	}

	t_octree_key ks[8];
	GetKeys(hashMask.dev_keys[index],hashMask,ks);

	t_index offsets[8];
	for (int i = 0;i < 8;i++) {
		offsets[i] = GetOffsetOfKey(ks[i],hashMask);
	}

#ifdef COMPUTE35
	t_double weight = __ldg(&hashMask.dev_weights[index]);
#else
	t_double weight = hashMask.dev_weights[index];
#endif
	for (int i = 0;i < 8;i++) {
		AtomicAdd(&x[offsets[i]],smat*weight);
		AtomicAdd(&x[offsets[i] + 1],smat*weight);
		AtomicAdd(&x[offsets[i] + 2],smat*weight);
	}
}

void GPUGridHash::MatrixVectorDiaK(t_double *dev_x,t_double smat,cudaStream_t stream) {
	int blockcount = _applyStarts/THREADS_PER_BLOCK + 1;
	MatrixVectorDiaHashKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(_applyStarts,_hashMask,dev_x,smat);
}

__global__ void MatrixVectorDiaHashCompKernel(t_index numStarts,HashMask hashMask,t_double *x,t_double smat) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}

	t_octree_key ks[8];
	GetKeys(hashMask.dev_keys[index],hashMask,ks);

	t_index offsets[8];
	for (int i = 0;i < 8;i++) {
		offsets[i] = GetOffsetOfKey(ks[i],hashMask);
	}

#ifdef COMPUTE35
	t_double weight = __ldg(&hashMask.dev_weights[index]);
#else
	t_double weight = hashMask.dev_weights[index];
#endif
	for (int i = 0;i < 8;i++) {
		AtomicAdd(&x[offsets[i]/3],smat*weight);
	}
}

void GPUGridHash::MatrixVectorDiaCompK(t_double *dev_x,t_double smat,cudaStream_t stream) {
	int blockcount = _applyStarts/THREADS_PER_BLOCK + 1;
	MatrixVectorDiaHashCompKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(_applyStarts,_hashMask,dev_x,smat);
}

__global__ void MatrixVectorDiaOffsetKernel(t_index numStarts,ElementOffsets elementOffsets,t_double *x,t_double smat) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}

	t_index offsets[8];
	GetOffsetsFromVector(index,elementOffsets,offsets);

#ifdef COMPUTE35
	t_double weight = __ldg(&elementOffsets.dev_weights[index]);
#else
	t_double weight = elementOffsets.dev_weights[index];
#endif
	for (int i = 0;i < 8;i++) {
		AtomicAdd(&x[offsets[i]],smat*weight);
		AtomicAdd(&x[offsets[i] + 1],smat*weight);
		AtomicAdd(&x[offsets[i] + 2],smat*weight);
	}
}

void GPUGridOffset::MatrixVectorDiaK(t_double *dev_x,t_double smat,cudaStream_t stream) {
	int blockcount = _applyStarts/THREADS_PER_BLOCK + 1;
	MatrixVectorDiaOffsetKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(_applyStarts,_elementOffsets,dev_x,smat);
}

__global__ void MatrixVectorDiaOffsetCompKernel(t_index numStarts,ElementOffsets elementOffsets,t_double *x,t_double smat) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}

	t_index offsets[8];
	GetOffsetsFromVector(index,elementOffsets,offsets);

#ifdef COMPUTE35
	t_double weight = __ldg(&elementOffsets.dev_weights[index]);
#else
	t_double weight = elementOffsets.dev_weights[index];
#endif
	for (int i = 0;i < 8;i++) {
		AtomicAdd(&x[offsets[i]/3],smat*weight);
	}
}

void GPUGridOffset::MatrixVectorDiaCompK(t_double *dev_x,t_double smat,cudaStream_t stream) {
	int blockcount = _applyStarts/THREADS_PER_BLOCK + 1;
	MatrixVectorDiaOffsetCompKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(_applyStarts,_elementOffsets,dev_x,smat);
}

__global__ void RestrictHashKernel(t_index numKeys,HashMask hashMaskCoarse,HashMask hashMaskFine,t_double *coarse,const t_double *__restrict__ fine) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numKeys) {
		return;
	}

	t_index fineindex = GetOffsetOfKey(hashMaskFine.dev_keys[index],hashMaskFine);

	t_octree_key coarsekey = hashMaskFine.dev_keys[index] >> 3;
	unsigned short LocalFineCoordinate = hashMaskFine.dev_keys[index] & 7;

	t_octree_key tmp = (coarsekey | hashMaskFine.maskx) + 1;
	t_octree_key diffx = (tmp&~hashMaskCoarse.maskx) | (coarsekey&hashMaskCoarse.maskx);
	diffx = diffx ^ coarsekey;
	tmp = (coarsekey | hashMaskCoarse.masky) + 1;
	t_octree_key diffy = (tmp&~hashMaskCoarse.masky) | (coarsekey&hashMaskCoarse.masky);
	diffy = diffy ^ coarsekey;
	tmp = (coarsekey | hashMaskCoarse.maskz) + 1;
	t_octree_key diffz = (tmp&~hashMaskCoarse.maskz) | (coarsekey&hashMaskCoarse.maskz);
	diffz = diffz ^ coarsekey;

	int max_x = (LocalFineCoordinate & 1);
	int max_y = (LocalFineCoordinate & 2);
	int max_z = (LocalFineCoordinate & 4);

	t_double factor = ((max_z >> 2) + 1) * ((max_y >> 1) + 1) * (max_x  + 1);
	factor = 1.0/(factor);//*8);

	t_octree_key new_coarsekey;
	t_index new_coarseindex;
	for(int z =0; z <= max_z; z+=4) {
		for(int y =0; y <= max_y; y+=2) {
			for(int x =0; x <= max_x; x++) {
				new_coarsekey = coarsekey;
				if (z!=0)
					new_coarsekey = new_coarsekey ^ diffz;
				if (y!=0)
					new_coarsekey = new_coarsekey ^ diffy;
				if (x!=0)
					new_coarsekey = new_coarsekey ^ diffx;

				new_coarseindex = GetOffsetOfKey(new_coarsekey,hashMaskCoarse);

				for(int i = 0;i < 3; i++) {
					AtomicAdd(&coarse[new_coarseindex + i],fine[fineindex + i]*factor);
				}
			}
		}
	}

}

void GPUGridHash::RestrictK(t_double *dev_coarse,t_double *dev_fine,cudaStream_t stream) {
	int blockcount = _rpStarts/THREADS_PER_BLOCK + 1;
	RestrictHashKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(_rpStarts,_hashMask,*_hashMask.hashMaskFine,dev_coarse,dev_fine);
}

__global__ void ProlongateHashKernel(t_index numKeys,HashMask hashMaskCoarse,HashMask hashMaskFine,const t_double *__restrict__ coarse,t_double *fine) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numKeys) {
		return;
	}

	t_index fineindex = GetOffsetOfKey(hashMaskFine.dev_keys[index],hashMaskFine);

	t_octree_key coarsekey = hashMaskFine.dev_keys[index] >> 3;
	unsigned short LocalFineCoordinate = hashMaskFine.dev_keys[index] & 7;



	t_octree_key tmp = (coarsekey | hashMaskFine.maskx) + 1;
	t_octree_key diffx = (tmp&~hashMaskFine.maskx) | (coarsekey&hashMaskFine.maskx);
	diffx = diffx ^ coarsekey;
	tmp = (coarsekey | hashMaskFine.masky) + 1;
	t_octree_key diffy = (tmp&~hashMaskFine.masky) | (coarsekey&hashMaskFine.masky);
	diffy = diffy ^ coarsekey;
	tmp = (coarsekey | hashMaskFine.maskz) + 1;
	t_octree_key diffz = (tmp&~hashMaskFine.maskz) | (coarsekey&hashMaskFine.maskz);
	diffz = diffz ^ coarsekey;

	int max_x = (LocalFineCoordinate & 1);
	int max_y = (LocalFineCoordinate & 2);
	int max_z = (LocalFineCoordinate & 4);

	t_double factor = ((max_z >> 2) + 1) * ((max_y >> 1) + 1) * (max_x  + 1);
	factor = 1.0/ (factor);//*8);

	t_octree_key new_coarsekey;
	t_index new_coarseindex;
	for(int z =0; z <= max_z; z+=4) {
		for(int y =0; y <= max_y; y+=2) {
			for(int x =0; x <= max_x; x++) {
				new_coarsekey = coarsekey;
				if (z!=0)
					new_coarsekey = new_coarsekey ^ diffz;
				if (y!=0)
					new_coarsekey = new_coarsekey ^ diffy;
				if (x!=0)
					new_coarsekey = new_coarsekey ^ diffx;

				new_coarseindex = GetOffsetOfKey(new_coarsekey,hashMaskCoarse);

				for(int i = 0;i < 3; i++) {
					AtomicAdd(&fine[fineindex  + i],coarse[new_coarseindex + i]*factor);
				}
			}
		}
	}

}

void GPUGridHash::ProlongateK(t_double *dev_coarse,t_double *dev_fine,cudaStream_t stream) {
	int blockcount = _rpStarts/THREADS_PER_BLOCK + 1;
	ProlongateHashKernel<<<blockcount,THREADS_PER_BLOCK,0,stream>>>(_rpStarts,_hashMask,*_hashMask.hashMaskFine,dev_coarse,dev_fine);
}

__global__ void RestrictOffsetKernel(t_index numStarts,ElementOffsets elementOffsets,t_double *coarse,const t_double *__restrict__ fine) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= numStarts) {
		return;
	}
#ifdef COMPUTE35
	t_index fineStartIndex = __ldg(&elementOffsets.dev_fineStartInds[index]);
#else
	t_index fineStartIndex = elementOffsets.dev_fineStartInds[index];
#endif
	if (fineStartIndex == NOFINEKEYS) {
		return;
	}

	t_index offsets[8];
	uchar localInds = GetOffsetsFromVectorRP(index,elementOffsets,offsets);

	Off8 lu = rpLookupTable[threadIdx.y];
	uchar num = lu.keys[0];

	for (int i = 0;i < threadIdx.y;i++) {
		if (localInds & 1) {
			fineStartIndex++;
		}
		localInds = localInds >> 1;
	}
	if (!(localInds & 1)) {
		return;
	}

	if (num < 8) {
		for (int i = 1;i < num;i++) {
			offsets[i] = offsets[lu.keys[i]];
		}
	}

	fineStartIndex *= 3;
	t_double factor = 1.0/num;
	t_double fineVals[3];

	fineVals[0] = fine[fineStartIndex]*factor;
	fineVals[1] = fine[fineStartIndex + 1]*factor;
	fineVals[2] = fine[fineStartIndex + 2]*factor;
	for (int i = 0;i < num;i++) {
		for (int j = 0;j < 3;j++) {
			AtomicAdd(&coarse[offsets[i] + j],fineVals[j]);
		}
	}
}

void GPUGridOffset::RestrictK(t_double *dev_coarse,t_double *dev_fine,cudaStream_t stream) {
	dim3 dim(32,8,1);
	int blockcount = _rpStarts/32 + 1;
	RestrictOffsetKernel<<<blockcount,dim,0,stream>>>(_rpStarts,_elementOffsets,dev_coarse,dev_fine);
}

__global__ void ProlongateOffsetKernel(t_index numStarts,ElementOffsets elementOffsets,const t_double *__restrict__ coarse,t_double *fine) {
	int index = threadIdx.y + blockIdx.x*blockDim.y;
	if (index >= numStarts) {
		return;
	}
#ifdef COMPUTE35
	t_index fineStartIndex = __ldg(&elementOffsets.dev_fineStartInds[index]);
#else
	t_index fineStartIndex = elementOffsets.dev_fineStartInds[index];
#endif
	if (fineStartIndex == NOFINEKEYS) {
		return;
	}

	t_index offsets[8];
	uchar localInds = GetOffsetsFromVectorRP(index,elementOffsets,offsets);

	Off8 lu = rpLookupTable[threadIdx.x];
	uchar num = lu.keys[0];

	for (int i = 0;i < threadIdx.x;i++) {
		if (localInds & 1) {
			fineStartIndex++;
		}
		localInds = localInds >> 1;
	}
	if (!(localInds & 1)) {
		return;
	}

	if (num < 8) {
		for (int i = 1;i < num;i++) {
			offsets[i] = offsets[lu.keys[i]];
		}
	}

	fineStartIndex *= 3;
	t_double factor = 1.0/num;
	t_double fineRes[3];
	fineRes[0] = 0;
	fineRes[1] = 0;
	fineRes[2] = 0;

	for (int i = 0;i < num;i++) {
		for (int j = 0;j < 3;j++) {
			fineRes[j] += coarse[offsets[i] + j]*factor;
		}
	}
	fine[fineStartIndex] += fineRes[0];
	fine[fineStartIndex + 1] += fineRes[1];
	fine[fineStartIndex + 2] += fineRes[2];
}

void GPUGridOffset::ProlongateK(t_double *dev_coarse,t_double *dev_fine,cudaStream_t stream) {
	dim3 dim(8,32,1);
	int blockcount = _rpStarts/32 + 1;
	ProlongateOffsetKernel<<<blockcount,dim,0,stream>>>(_rpStarts,_elementOffsets,dev_coarse,dev_fine);
}

#define ACCESS(r,c,rows,cols)\
    (c)*(rows) + (r)

#define nst 6
#define nip 1
#define ndim 3
#define nod 8
#define ndof 24

void __device__ d_mat(double *d ,double e, double v) {
    //double v1;
    double v2,c,vv;
    int i;
    //Write zeros in d
    for (i=0; i< nst*nst; i++)
        d[i]=0.0;

    //v1 = 1.0 - v;
    v2 = v/(1.0-v);
    vv = (1.0-2.0*v)/(1.0-v)*0.5;
    //set diag 1-3 to 1
    for (i =0; i<3;i++)
        d[ACCESS(i,i,nst,nst)] = 1.0;
    //set diag 4-6 to (1-2v)/(2(1-v))
    for (i =3; i<6;i++)
            d[ACCESS(i,i,nst,nst)] = vv;

    d[ACCESS(0,1,nst,nst)] = v2;
    d[ACCESS(1,0,nst,nst)] = v2;
    d[ACCESS(0,2,nst,nst)] = v2;
    d[ACCESS(2,0,nst,nst)] = v2;
    d[ACCESS(1,2,nst,nst)] = v2;
    d[ACCESS(2,1,nst,nst)] = v2;

    c = e/(2.0*(1.0+v)*vv);
    for (i=0; i< nst*nst; i++)
        d[i]*=c;

}

void inline __device__ sample(double *s, double *wt){
    //only hexahedron
    double b=0,c=0,root3=0,r15=0;
    double w[3],v[9];
    int i;

    //root3 = 1.0/sqrt(3.0);
    root3=0.57735026918962576452;
    //r15 = 0.2*sqrt(15.0);
    r15= 0.77459666924148337704;
    w[0]=5.0/9.0;
    w[1]=8.0/9.0;
    w[2]=5.0/9.0;
    for (int j= 0; j <3; j++) {
        for (i =0; i<3; i++) {
            v[3*j +i]=w[j]*w[i];
        }
    }

	switch (nip)
	{
		case 1:
			s[0]= 0.0;
			s[1]= 0.0;
			s[2]=0.0;
			wt[0] = 8.0;
			break;
		case 8:
			s[ACCESS(0,0,nip,3)] = root3;
			s[ACCESS(0,1,nip,3)] = root3;
			s[ACCESS(0,2,nip,3)] = root3;
			s[ACCESS(1,0,nip,3)] = root3;
			s[ACCESS(1,1,nip,3)] = root3;
			s[ACCESS(1,2,nip,3)] = -root3;
			s[ACCESS(2,0,nip,3)] = root3;
			s[ACCESS(2,1,nip,3)] = -root3;
			s[ACCESS(2,2,nip,3)] = root3;
			s[ACCESS(3,0,nip,3)] = root3;
			s[ACCESS(3,1,nip,3)] = -root3;
			s[ACCESS(3,2,nip,3)] = -root3;
			s[ACCESS(4,0,nip,3)] = -root3;
			s[ACCESS(4,1,nip,3)] = root3;
			s[ACCESS(4,2,nip,3)] = root3;
			s[ACCESS(5,0,nip,3)] = -root3;
			s[ACCESS(5,1,nip,3)] = -root3;
			s[ACCESS(5,2,nip,3)] = root3;
			s[ACCESS(6,0,nip,3)] = -root3;
			s[ACCESS(6,1,nip,3)] = root3;
			s[ACCESS(6,2,nip,3)] = -root3;
			s[ACCESS(7,0,nip,3)] = -root3;
			s[ACCESS(7,1,nip,3)] = -root3;
			s[ACCESS(7,2,nip,3)] = -root3;
			for ( i = 0; i < nip; i++)
				wt[i] = 1.0;
			break;
		case 14:
			b = 0.795822426;
			c = 0.758786911;
			for  (i = 0; i < nip; i++) {
				if (7 > i)
					wt[i]=0.886426593;
				else
					wt[i] = 0.335180055;
			}

			s[ACCESS(0,0,nip,3)] = -b;
			s[ACCESS(1,0,nip,3)] = b;
			s[ACCESS(2,1,nip,3)] = -b;
			s[ACCESS(3,1,nip,3)] = b;
			s[ACCESS(4,2,nip,3)] = -b;
			s[ACCESS(5,2,nip,3)] = b;

			for  (i = 6; i < nip; i++) {
				s[ACCESS(i,0,nip,3)] = c;
				s[ACCESS(i,1,nip,3)] = c;
				s[ACCESS(i,2,nip,3)] = c;
			}
			s[ACCESS(6,0,nip,3)] = -c;
			s[ACCESS(6,1,nip,3)] = -c;
			s[ACCESS(6,2,nip,3)] = -c;
			s[ACCESS(7,1,nip,3)] = -c;
			s[ACCESS(7,2,nip,3)] = -c;
			s[ACCESS(8,0,nip,3)] = -c;
			s[ACCESS(8,2,nip,3)] = -c;
			s[ACCESS(9,2,nip,3)] = -c;
			s[ACCESS(10,0,nip,3)] = -c;
			s[ACCESS(10,1,nip,3)] = -c;
			s[ACCESS(11,1,nip,3)] = -c;
			s[ACCESS(12,0,nip,3)] = -c;
			break;
		case 15:
			b = 1.0;
			c = 0.674199862;
			wt[0] = 1.564444444;
			for  (i = 1; i < nip; i++) {
				if (8 > i)
					wt[i]=0.355555556;
				else
					wt[i] = 0.537777778;
			}
			s[ACCESS(1,0,nip,3)] = -b;
			s[ACCESS(2,0,nip,3)] = b;
			s[ACCESS(3,1,nip,3)] = -b;
			s[ACCESS(4,1,nip,3)] = b;
			s[ACCESS(5,2,nip,3)] = -b;
			s[ACCESS(6,2,nip,3)] = b;
			for  (i = 7; i < nip; i++) {
				s[ACCESS(i,0,nip,3)] = c;
				s[ACCESS(i,1,nip,3)] = c;
				s[ACCESS(i,2,nip,3)] = c;
			}
			s[ACCESS(7,0,nip,3)] = -c;
			s[ACCESS(7,1,nip,3)] = -c;
			s[ACCESS(7,2,nip,3)] = -c;
			s[ACCESS(8,1,nip,3)] = -c;
			s[ACCESS(8,2,nip,3)] = -c;
			s[ACCESS(9,0,nip,3)] = -c;
			s[ACCESS(9,2,nip,3)] = -c;
			s[ACCESS(10,2,nip,3)] = -c;
			s[ACCESS(11,0,nip,3)] = -c;
			s[ACCESS(11,1,nip,3)] = -c;
			s[ACCESS(12,1,nip,3)] = -c;
			s[ACCESS(13,0,nip,3)] = -c;
			break;
		case 27:
			for (int j= 0; j <3; j++) {
				for (i =0; i<9; i++) {
					wt[9*j +i]=w[j]*v[i];
				}
			}
			for  (i = 0; i < 25; i=i+3)
				s[ACCESS(i,0,nip,3)] = -r15;
			for  (i = 1; i < 26; i=i+3)
				s[ACCESS(i,0,nip,3)] = 0.0;
			for  (i = 2; i < 27; i=i+3)
					s[ACCESS(i,0,nip,3)] = r15;

			for (int j=0; j < 27;j=j+9) {
				for  (i = j; i < j+3; i++)
					s[ACCESS(i,2,nip,3)] = r15;
				for  (i = j+3; i < j+6; i++)
					s[ACCESS(i,2,nip,3)] = 0;
				for  (i = j+6;i < j+9; i++)
					s[ACCESS(i,2,nip,3)] = -r15;
			}

			for  (i = 0; i < 9; i++)
				s[ACCESS(i,1,nip,3)] = -r15;
			for  (i = 9; i < 18; i++)
				s[ACCESS(i,1,nip,3)] = 0;
			for  (i = 18; i < 27; i++)
				s[ACCESS(i,1,nip,3)] = r15;
			break;

	}
    return;
}

void inline __device__ shape_der(double * der, double *points, int ipoint){
    double xi, eta, zeta, etam,xim,zetam,xip,etap,zetap;
    xi = points[ACCESS(ipoint,0,nip,ndim)];
    eta = points[ACCESS(ipoint,1,nip,ndim)];
    zeta = points[ACCESS(ipoint,2,nip,ndim)];
    etam = 1.0-eta;
    xim = 1.0-xi;
    zetam = 1.0-zeta;
    etap = eta+1.0;
    xip = xi+1.0;
    zetap = zeta+1.0;

    der[ACCESS(0,0,ndim,nod)] = -0.125 *etam *zetam;
    der[ACCESS(0,1,ndim,nod)] = -0.125 *etam *zetap;
    der[ACCESS(0,2,ndim,nod)] = 0.125 *etam *zetap;
    der[ACCESS(0,3,ndim,nod)] = 0.125 *etam *zetam;
    der[ACCESS(0,4,ndim,nod)] = -0.125 *etap *zetam;
    der[ACCESS(0,5,ndim,nod)] = -0.125 *etap *zetap;
    der[ACCESS(0,6,ndim,nod)] = 0.125 *etap *zetap;
    der[ACCESS(0,7,ndim,nod)] = 0.125 *etap *zetam;

    der[ACCESS(1,0,ndim,nod)] = -0.125 *xim *zetam;
    der[ACCESS(1,1,ndim,nod)] = -0.125 *xim *zetap;
    der[ACCESS(1,2,ndim,nod)] = -0.125 *xip *zetap;
    der[ACCESS(1,3,ndim,nod)] = -0.125 *xip *zetam;
    der[ACCESS(1,4,ndim,nod)] = 0.125 *xim *zetam;
    der[ACCESS(1,5,ndim,nod)] = 0.125 *xim *zetap;
    der[ACCESS(1,6,ndim,nod)] = 0.125 *xip *zetap;
    der[ACCESS(1,7,ndim,nod)] = 0.125 *xip *zetam;

    der[ACCESS(2,0,ndim,nod)] = -0.125 *xim *etam;
    der[ACCESS(2,1,ndim,nod)] = 0.125 *xim *etam;
    der[ACCESS(2,2,ndim,nod)] = 0.125 *xip *etam;
    der[ACCESS(2,3,ndim,nod)] = -0.125 *xip *etam;
    der[ACCESS(2,4,ndim,nod)] = -0.125 *xim *etap;
    der[ACCESS(2,5,ndim,nod)] = 0.125 *xim *etap;
    der[ACCESS(2,6,ndim,nod)] = 0.125 *xip *etap;
    der[ACCESS(2,7,ndim,nod)] = -0.125 *xip *etap;

    return;
}

//matmult with the second matrix transposed
int inline __device__ matmulABt(const double *A, int Ar, int Ac, const double *B, int Br, int Bc, double *C, int Cr, int Cc){
    double sum;
    int i,j,k;
//     if ((Ac != Bc) || (Ar != Cr) ||(Br != Cc)) {
//         cout << "*********************ERROR DIMENSION OF THE MATRIX NOT OKEY**************** \n";
//         return (-1);
//     }

    for(i=0; i<Ar; i++)
        for(j=0; j<Br; j++) {
        sum=0;
        for(k=0; k<Bc; k++)
            sum += A[ACCESS(i,k,Ar,Ac)]*B[ACCESS(j,k,Br,Bc)];
        C[ACCESS(i,j,Cr,Cc)] = sum;
        }

        return (0);
}

double inline __device__ determinant(double* a,int n) {
    if ( 1 == n)
        return a[0];
    else if ( 2 == n )
        return (a[ACCESS(0,0,2,2)]*a[ACCESS(1,1,2,2)] -a[ACCESS(1,0,2,2)]*a[ACCESS(0,1,2,2)]);
    else if ( 3 == n) {//von Saurrus
        return (a[ACCESS(0,0,3,3)]*a[ACCESS(1,1,3,3)]*a[ACCESS(2,2,3,3)]
                +a[ACCESS(0,1,3,3)]*a[ACCESS(1,2,3,3)]*a[ACCESS(2,0,3,3)]
                +a[ACCESS(0,2,3,3)]*a[ACCESS(1,0,3,3)]*a[ACCESS(2,1,3,3)]
                -a[ACCESS(0,2,3,3)]*a[ACCESS(1,1,3,3)]*a[ACCESS(2,0,3,3)]
                -a[ACCESS(0,0,3,3)]*a[ACCESS(1,2,3,3)]*a[ACCESS(2,1,3,3)]
                -a[ACCESS(0,1,3,3)]*a[ACCESS(1,0,3,3)]*a[ACCESS(2,2,3,3)]);
    }
    else {
        return -666.666;
    }
}

void inline __device__ invert(double *a, double *b, int n){
    double idet = 1/determinant(a,n);
    if ( 1 == n) {
        b[0]=1/a[0];
        return;
    }
    else if ( 2 == n)
    {
        return;
    }
    else if ( 3 == n) {
        b[ACCESS(0,0,3,3)] = idet * (a[ACCESS(1,1,3,3)]*a[ACCESS(2,2,3,3)] - a[ACCESS(2,1,3,3)]*a[ACCESS(1,2,3,3)]);
        b[ACCESS(1,0,3,3)] = idet * (a[ACCESS(2,0,3,3)]*a[ACCESS(1,2,3,3)] - a[ACCESS(1,0,3,3)]*a[ACCESS(2,2,3,3)]);
        b[ACCESS(2,0,3,3)] = idet * (a[ACCESS(1,0,3,3)]*a[ACCESS(2,1,3,3)] - a[ACCESS(2,0,3,3)]*a[ACCESS(1,1,3,3)]);
        b[ACCESS(0,1,3,3)] = idet * (a[ACCESS(2,1,3,3)]*a[ACCESS(0,2,3,3)] - a[ACCESS(0,1,3,3)]*a[ACCESS(2,2,3,3)]);
        b[ACCESS(1,1,3,3)] = idet * (a[ACCESS(0,0,3,3)]*a[ACCESS(2,2,3,3)] - a[ACCESS(2,0,3,3)]*a[ACCESS(0,2,3,3)]);
        b[ACCESS(2,1,3,3)] = idet * (a[ACCESS(2,0,3,3)]*a[ACCESS(0,1,3,3)] - a[ACCESS(0,0,3,3)]*a[ACCESS(2,1,3,3)]);
        b[ACCESS(0,2,3,3)] = idet * (a[ACCESS(0,1,3,3)]*a[ACCESS(1,2,3,3)] - a[ACCESS(1,1,3,3)]*a[ACCESS(0,2,3,3)]);
        b[ACCESS(1,2,3,3)] = idet * (a[ACCESS(1,0,3,3)]*a[ACCESS(0,2,3,3)] - a[ACCESS(0,0,3,3)]*a[ACCESS(1,2,3,3)]);
        b[ACCESS(2,2,3,3)] = idet * (a[ACCESS(0,0,3,3)]*a[ACCESS(1,1,3,3)] - a[ACCESS(1,0,3,3)]*a[ACCESS(0,1,3,3)]);
        return;
    }
}

void inline __device__ b_mat(double *B, double *deriv) {
    int l,k,n;
    double x,y,z;
    for (int i = 0;i < nst * ndof; i++)
        B[i]=0;

    for (int i = 0; i < nod ; i++) {
        l = 3*i;
        k = l+1;
        n = l+2;
        x = deriv[ACCESS(0,i,3,nod)];
        y = deriv[ACCESS(1,i,3,nod)];
        z = deriv[ACCESS(2,i,3,nod)];//TODO: to check the row
        B[ACCESS(0,l,6,ndof)] = x;
        B[ACCESS(3,k,6,ndof)] = x;
        B[ACCESS(5,n,6,ndof)] = x;
        B[ACCESS(1,k,6,ndof)] = y;
        B[ACCESS(3,l,6,ndof)] = y;
        B[ACCESS(4,n,6,ndof)] = y;
        B[ACCESS(2,n,6,ndof)] = z;
        B[ACCESS(4,k,6,ndof)] = z;
        B[ACCESS(5,l,6,ndof)] = z;
    }
}

int  inline __device__ matmulAB(const double *A, int Ar, int Ac, const double *B, int Br, int Bc, double *C, int Cr, int Cc){
    double sum;
    int i,j,k;
//     if ((Ac != Br) || (Ar != Cr) ||(Bc != Cc)) {
//         cout << "*********************ERROR DIMENSION OF THE MATRIX NOT OKEY**************** \n";
//         return (-1);
//     }

    for(i=0; i<Ar; i++)
        for(j=0; j<Bc; j++) {
        sum=0;
        for(k=0; k<Br; k++)
            sum += A[ACCESS(i,k,Ar,Ac)]*B[ACCESS(k,j,Br,Bc)];
        C[ACCESS(i,j,Cr,Cc)] = sum;
        }

        return (0);
}

void inline __device__ invar(double *stress, double &sigma, double &dsbar, double &theta) {
    double sx,sy,sz,txy,dx,dy,dz,xj3,sine,s1,s2,s3,s4,s5,s6,ds1,ds2,ds3,d2,d3,sq3;
    switch (nst) {
        case 4:
            sx = stress[0];
            sy = stress[1];
            txy = stress[2];
            sz = stress[3];
            sigma = (sx+sy+sz)/3.0;
            dsbar = sqrt((sx-sy)*(sx-sy)+(sy-sz)*(sy-sz)+(sz-sx)*(sz-sx)+6.0*txy*txy)/sqrt(2.0);
            if (dsbar<1.e-10)
                theta = .0;
            else
            {
                dx = (2.0*sx-sy-sz)/3.0;
                dy = (2.0*sy-sz-sx)/3.0;
                dz = (2.0*sz-sx-sy)/3.0;
                xj3 = dx*dy*dz-dz*txy*txy;
                sine = -13.5*xj3/(dsbar*dsbar*dsbar);
                if (sine>1.)
                    sine = 1.0;
                if (sine<-1.)
                    sine = -1.0;
                theta = asin(sine)/3.0;
            }
            break;
        case 6:
            sq3 = sqrt(3.0);
            s1 = stress[0];
            s2 = stress[1];
            s3 = stress[2];
            s4 = stress[3];
            s5 = stress[4];
            s6 = stress[5];
            sigma = (s1+s2+s3)/3.0;
            d2 = ((s1-s2)*(s1-s2)+(s2-s3)*(s2-s3)+(s3-s1)*(s3-s1))/6.0+s4*s4+s5*s5+s6*s6;
            ds1 = s1-sigma;
            ds2 = s2-sigma;
            ds3 = s3-sigma;
            d3 = ds1*ds2*ds3-ds1*s5*s5-ds2*s6*s6-ds3*s4*s4+2.0*s4*s5*s6;
            dsbar = sq3*sqrt(d2);
            if (dsbar == 0.)
                theta = 0.0;
            else {
                sine = -3.0*sq3*d3/(2.0*d2*sqrt(d2));
                if (sine>1.)
                    sine = 1.0;
                if (sine<-1.)
                    sine = -1.0;
                theta = asin(sine)/3.0;
            }
            break;
    }
    return;
}

__global__ void PostProcessHashKernel(t_index numStarts,HashMask hashMask,t_double *g_num,t_double *x,t_double *y,t_double *strainOut,
                                      t_double poisson_ratio) {
	//nip = 1;
	//nst = 6;
	//ndim = 3;
	//nod = 8;
	//ndof = 24;

	int index = threadIdx.x + blockIdx.x*blockDim.x;
	t_index offsets[8];
	t_double disps[24];
	t_double e;

	if (index >= numStarts) {
		return;
	}

	t_octree_key ks[8];
	GetKeys(hashMask.dev_keys[index],hashMask,ks);

	for (int i = 0;i < 8;i++) {

		offsets[i] = GetOffsetOfKey(ks[i],hashMask);
		disps[i*3] = x[offsets[i]];
		disps[i*3 + 1] = x[offsets[i] + 1];
		disps[i*3 + 2] = x[offsets[i] + 2];
	}
	e = hashMask.dev_weights[index]*1000;

	t_double strain[(nst + 1)*nip];
	t_double stress[(nst + 1)*nip];

	for (int i = 0;i < (nst + 1)*nip;i++) {
		strain[i] = 0.0;
		stress[i] = 0.0;
	}

	t_double D[nst*nst];
	d_mat(D,e,poisson_ratio);

	t_double points[nip*ndim];
	t_double weights[nip];
	sample(points,weights);

	t_double der[ndim*nod];
	t_double jac1[ndim*ndim];
	t_double jac[ndim*ndim];
	t_double deriv[ndim*nod];
	t_double B[nst*ndof];
	t_double Bld[nst];
	t_double deeld[nst];
	t_double sigma[nip];
	t_double theta[nip];
	for (int i = 0;i < nip;i++) {
		shape_der (der,points,i);
		matmulABt(der,ndim,nod, g_num,ndim,nod, jac1,ndim,ndim);
		//det = determinant(jac1, ndim);
		invert(jac1,jac,ndim);
		matmulAB(jac,ndim,ndim, der,ndim,nod, deriv,ndim,nod);
		b_mat(B,deriv);
		matmulAB(B,nst,ndof,disps,ndof,1,Bld,nst,1);
		for (int n = 0;n < nst;n++) {
			strain[ACCESS(n,i,nst+1,nip)]=Bld[n];
		}
		matmulAB(D,nst,nst,Bld,nst,1,deeld,nst,1);
		for (int n = 0;n < nst;n++) {
			stress[ACCESS(n,i,nst+1,nip)]=deeld[n];
		}
		invar(deeld, sigma[i], stress[ACCESS(nst,i,nst+1,nip)], theta[i]);
		for(int j = 0;j < nst;j++) {
			strain[ACCESS(nst,i,nst+1,nip)] = strain[ACCESS(nst,i,nst+1,nip)] + strain[ACCESS(j,i,nst+1,nip)]*stress[ACCESS(j,i,nst+1,nip)];
		}
		strain[ACCESS(nst,i,nst+1,nip)]*=0.5;
	}

	y[offsets[0]] = stress[(nst + 1)*nip - 1];
	y[offsets[0] + 1] = strain[(nst + 1)*nip - 1];

	// Full strain tensor (6 Voigt components; nip=1 so the Gauss-point index
	// is always 0, i.e. strain[n] == strain[ACCESS(n,0,nst+1,nip)]) into the
	// 6-wide-per-node-slot buffer allocated in Postprocessing.h. Base index
	// doubled vs. y's 3-wide indexing to match dev_strain's 2x size.
	t_index strainBase = 2*offsets[0];
	for (int n = 0;n < nst;n++) {
		strainOut[strainBase + n] = strain[n];
	}

}

void GPUGridHash::PostProcessK(t_double *dev_x, t_double *dev_y, t_double *g_num, t_double *dev_strain,
                               t_double poisson_ratio, cudaStream_t stream) {
	int blockcount = _applyStarts/128 + 1;
	PostProcessHashKernel<<<blockcount,128,0,stream>>>(_applyStarts,_hashMask,g_num,dev_x,dev_y,dev_strain,poisson_ratio);
}

__global__ void PostProcessOffsetKernel(t_index numStarts,ElementOffsets elementOffsets,t_double *g_num,t_double *x,t_double *y,t_double *strainOut,
                                      t_double poisson_ratio) {
	//nip = 1;
	//nst = 6;
	//ndim = 3;
	//nod = 8;
	//ndof = 24;

	int index = threadIdx.x + blockIdx.x*blockDim.x;
	t_index offsets[8];
	t_double disps[24];
	t_double e;

	if (index >= numStarts) {
		return;
	}

	GetOffsetsFromVector(index,elementOffsets,offsets);
	for (int i = 0;i < 8;i++) {
		disps[i*3] = x[offsets[i]];
		disps[i*3 + 1] = x[offsets[i] + 1];
		disps[i*3 + 2] = x[offsets[i] + 2];
	}
	e = elementOffsets.dev_weights[index]*1000;

	t_double strain[(nst + 1)*nip];
	t_double stress[(nst + 1)*nip];

	for (int i = 0;i < (nst + 1)*nip;i++) {
		strain[i] = 0.0;
		stress[i] = 0.0;
	}

	t_double D[nst*nst];
	d_mat(D,e,poisson_ratio);

	t_double points[nip*ndim];
	t_double weights[nip];
	sample(points,weights);

	t_double der[ndim*nod];
	t_double jac1[ndim*ndim];
	t_double jac[ndim*ndim];
	t_double deriv[ndim*nod];
	t_double B[nst*ndof];
	t_double Bld[nst];
	t_double deeld[nst];
	t_double sigma[nip];
	t_double theta[nip];
	for (int i = 0;i < nip;i++) {
		shape_der (der,points,i);
		matmulABt(der,ndim,nod, g_num,ndim,nod, jac1,ndim,ndim);
		//det = determinant(jac1, ndim);
		invert(jac1,jac,ndim);
		matmulAB(jac,ndim,ndim, der,ndim,nod, deriv,ndim,nod);
		b_mat(B,deriv);
		matmulAB(B,nst,ndof,disps,ndof,1,Bld,nst,1);
		for (int n = 0;n < nst;n++) {
			strain[ACCESS(n,i,nst+1,nip)]=Bld[n];
		}
		matmulAB(D,nst,nst,Bld,nst,1,deeld,nst,1);
		for (int n = 0;n < nst;n++) {
			stress[ACCESS(n,i,nst+1,nip)]=deeld[n];
		}
		invar(deeld, sigma[i], stress[ACCESS(nst,i,nst+1,nip)], theta[i]);
		for(int j = 0;j < nst;j++) {
			strain[ACCESS(nst,i,nst+1,nip)] = strain[ACCESS(nst,i,nst+1,nip)] + strain[ACCESS(j,i,nst+1,nip)]*stress[ACCESS(j,i,nst+1,nip)];
		}
		strain[ACCESS(nst,i,nst+1,nip)]*=0.5;
	}

	y[offsets[0]] = stress[(nst + 1)*nip - 1];
	y[offsets[0] + 1] = strain[(nst + 1)*nip - 1];

	t_index strainBase = 2*offsets[0];
	for (int n = 0;n < nst;n++) {
		strainOut[strainBase + n] = strain[n];
	}

}

void GPUGridOffset::PostProcessK(t_double *dev_x, t_double *dev_y, t_double *g_num, t_double *dev_strain,
                               t_double poisson_ratio, cudaStream_t stream) {
	int blockcount = _applyStarts/128 + 1;
	PostProcessOffsetKernel<<<blockcount,128,0,stream>>>(_applyStarts,_elementOffsets,g_num,dev_x,dev_y,dev_strain,poisson_ratio);
}
