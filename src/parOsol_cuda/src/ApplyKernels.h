/*
 * ApplyKernels.h
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

#ifndef APPLYKERNELS_H_
#define APPLYKERNELS_H_

#include <vector>
#include "GPUMemoryManager.h"
#include "GPUGridContainer.h"

void SetLocalStiffnessMatrix(t_double *sMat,int offset);
void SetRpLookupTable();
cudaError_t CheckCudaKernelImage();

class GPUGridOffset {

public:

	GPUGridOffset(GPUGridContainer gpuGridContainer,GPUMemoryManager *gpuMM) {
		_elementOffsets = gpuGridContainer.elementOffsets;
		_applyStarts = gpuGridContainer.applyStarts;
		_rpStarts = gpuGridContainer.rpStarts;
		_syncData = gpuGridContainer.syncData;
		_gpuMM = gpuMM;
	}

	~GPUGridOffset() {
		_gpuMM->FreeMemory(Grid,_syncData.level);
		CCS(cudaFree(_elementOffsets.dev_startInds),"~GPUGridOffset 1");
		CCS(cudaFree(_elementOffsets.dev_offsets_long1),"~GPUGridOffset 2");
		CCS(cudaFree(_elementOffsets.dev_offsets_long2),"~GPUGridOffset 3");
		CCS(cudaFree(_elementOffsets.dev_offsets_long3),"~GPUGridOffset 4");
		CCS(cudaFree(_elementOffsets.dev_offsets_last),"~GPUGridOffset 5");
		CCS(cudaFree(_elementOffsets.dev_weights),"~GPUGridOffset 6");
		if (_elementOffsets.dev_fineStartInds != 0) {
			CCS(cudaFree(_elementOffsets.dev_fineStartInds),"~GPUGridOffset 7");
			CCS(cudaFree(_elementOffsets.dev_localInds),"~GPUGridOffset 8");
		}
		_gpuMM->FreeMemory(Buffer,_syncData.level);

		if (_syncData.switching) {
			delete _syncData.cpuInds;
			CCS(cudaFree(_syncData.gpuVecA),"dsf");
			CCS(cudaFree(_syncData.gpuVecB),"dsf");
			_gpuMM->FreeMemory(DispVec,_syncData.level);
		} else {
			CCS(cudaFree(_syncData.gpuBuffer),"~GPUGridOffset 11");
			CCS(cudaFree(_syncData.gpuInds),"~GPUGridOffset 12");
			delete _syncData.cpuBuffer;
		}

	}

	void MatrixVectorK(t_double *dev_x,t_double *dev_y,t_double *dev_smat,cudaStream_t stream = 0);
	void MatrixVectorCK(t_double *dev_x,t_double *dev_y,int matOffset,cudaStream_t stream = 0);
	void MatrixVectorDiaK(t_double *dev_x,t_double smat,cudaStream_t stream = 0);
	void MatrixVectorDiaCompK(t_double *dev_x,t_double smat,cudaStream_t stream = 0);
	void RestrictK(t_double *dev_coarse,t_double *dev_fine,cudaStream_t stream = 0);
	void ProlongateK(t_double *dev_coarse,t_double *dev_fine,cudaStream_t stream = 0);

	void PostProcessK(t_double *dev_x, t_double *dev_y, t_double *g_num, t_double *dev_strain,
	                  t_double poisson_ratio, cudaStream_t stream = 0);

	ElementOffsets _elementOffsets;
	GPUMemoryManager *_gpuMM;
	t_index _applyStarts;
	t_index _rpStarts;

	SyncData _syncData;

};

class GPUGridHash {

public:

	GPUGridHash(GPUGridContainer gpuGridContainer,GPUMemoryManager *gpuMM) {
		_hashMask = gpuGridContainer.hashMask;
		_applyStarts = gpuGridContainer.applyStarts;
		_rpStarts = gpuGridContainer.rpStarts;
		_syncData = gpuGridContainer.syncData;
		_gpuMM = gpuMM;
	}

	~GPUGridHash() {
		_gpuMM->FreeMemory(Grid,_syncData.level);

		HashMaskGpu hashMasksGpu[NUMHASH];
		CCS(cudaMemcpy(hashMasksGpu,_hashMask.dev_hashMaskGpu,NUMHASH*sizeof(HashMaskGpu),cudaMemcpyDeviceToHost),"~GPUGridHash 1");

		for (int i = 0;i < NUMHASH;i++) {
			CCS(cudaFree(hashMasksGpu[i].dev_hashTable),"~GPUGridHash 2");
			CCS(cudaFree(hashMasksGpu[i].dev_offsetTable),"~GPUGridHash 3");
		}
		CCS(cudaFree(_hashMask.dev_hashMaskGpu),"~GPUGridHash 4");
		CCS(cudaFree(_hashMask.dev_keys),"~GPUGridHash 5");
		CCS(cudaFree(_hashMask.dev_weights),"~GPUGridHash 6");

		_gpuMM->FreeMemory(Buffer,_syncData.level);

		if (_syncData.switching) {
			delete _syncData.cpuInds;
			CCS(cudaFree(_syncData.gpuVecA),"dsf");
			CCS(cudaFree(_syncData.gpuVecB),"dsf");
			_gpuMM->FreeMemory(DispVec,_syncData.level);
		} else {
			CCS(cudaFree(_syncData.gpuBuffer),"~GPUGridOffset 11");
			CCS(cudaFree(_syncData.gpuInds),"~GPUGridOffset 12");
			delete _syncData.cpuBuffer;
		}

	}

	void SetGPUMemoryManager(GPUMemoryManager *gpuMM) {
		_gpuMM = gpuMM;
	}

	void MatrixVectorK(t_double *dev_x,t_double *dev_y,t_double *dev_smat,cudaStream_t stream = 0);
	void MatrixVectorCK(t_double *dev_x,t_double *dev_y,int matOffset,cudaStream_t stream = 0);
	void MatrixVectorDiaK(t_double *dev_x,t_double smat,cudaStream_t stream = 0);
	void MatrixVectorDiaCompK(t_double *dev_x,t_double smat,cudaStream_t stream = 0);
	void RestrictK(t_double *dev_coarse,t_double *dev_fine,cudaStream_t stream = 0);
	void ProlongateK(t_double *dev_coarse,t_double *dev_fine,cudaStream_t stream = 0);

	void PostProcessK(t_double *dev_x, t_double *dev_y, t_double *g_num, t_double *dev_strain,
	                  t_double poisson_ratio, cudaStream_t stream = 0);

	HashMask _hashMask;
	GPUMemoryManager *_gpuMM;
	t_index _applyStarts;
	t_index _rpStarts;

	SyncData _syncData;

};

#endif /* APPLYKERNELS_H_ */
