/*
 * GPUGridContainer.h
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

#ifndef GPUGRIDCONTAINER_H_
#define GPUGRIDCONTAINER_H_

#include "SyncData.h"

class GPUGridContainer {

public:

	ElementOffsets elementOffsets;
	HashMask hashMask;

	t_index applyStarts;
	t_index rpStarts;

	SyncData syncData;

#ifdef HASH

	HashMaskGpu hashMasksCpu[NUMHASH];
	HashMaskGpu hashMasksGpu[NUMHASH];

	t_octree_key *keys;
	t_index keysSize;

#else

	uint *startInds;
	t_index startIndsSize;

	offsetD *offsetsLong1;
	t_index offsetsLong1Size;

	offsetD *offsetsLong2;
	t_index offsetsLong2Size;

	offsetD *offsetsLong3;
	t_index offsetsLong3Size;

	uint *offsetsLast;
	t_index offsetsLastSize;

	t_index *fineStartInds;
	t_index fineStartIndsSize;

	uchar *localInds;
	t_index localIndsSize;
#endif

	t_double *weights;
	t_index weightsSize;

	t_index gpuBufferSize;

	t_index gpuIndsSize;

	long GetNumBytes() {
		long numBytes = 0;
#ifdef HASH
		for (int i = 0;i < NUMHASH;i++) {
			numBytes += hashMasksCpu[i].hashTableSize*sizeof(t_index);
			numBytes += hashMasksCpu[i].offsetTableSize*sizeof(t_offset);
		}
		numBytes += keysSize*sizeof(t_octree_key);
#else
		numBytes += startIndsSize*sizeof(uint);
		numBytes += offsetsLong1Size*sizeof(offsetD);
		numBytes += offsetsLong2Size*sizeof(offsetD);
		numBytes += offsetsLong3Size*sizeof(offsetD);
		numBytes += offsetsLastSize*sizeof(uint);
		numBytes += fineStartIndsSize*sizeof(t_index);
		numBytes += localIndsSize*sizeof(char);
#endif
		numBytes += weightsSize*sizeof(t_double);
		numBytes += gpuBufferSize*sizeof(t_double);
		numBytes += gpuIndsSize*sizeof(t_index);
		return numBytes;
	}

	void MoveDataToGpu(GPUMemoryManager *gpuMM) {
		long numBytes = 0;

#ifdef HASH

		for (int i = 0;i < NUMHASH;i++) {
			numBytes += hashMasksCpu[i].hashTableSize*sizeof(t_index);
			CCS(cudaMalloc((void**) &hashMasksGpu[i].dev_hashTable,hashMasksCpu[i].hashTableSize*sizeof(t_index)),"BuildGrid 15");
			CCS(cudaMemcpy(hashMasksGpu[i].dev_hashTable,hashMasksCpu[i].dev_hashTable,hashMasksCpu[i].hashTableSize*sizeof(t_index),cudaMemcpyHostToDevice),"BuildGrid 16");
			hashMasksGpu[i].hashTableSize = hashMasksCpu[i].hashTableSize;
			hashMasksGpu[i].hashTableMod = hashMasksCpu[i].hashTableMod;

			numBytes += hashMasksCpu[i].offsetTableSize*sizeof(t_offset);
			CCS(cudaMalloc((void**) &hashMasksGpu[i].dev_offsetTable,hashMasksCpu[i].offsetTableSize*sizeof(t_offset)),"BuildGrid 15");
			CCS(cudaMemcpy(hashMasksGpu[i].dev_offsetTable,hashMasksCpu[i].dev_offsetTable,hashMasksCpu[i].offsetTableSize*sizeof(t_offset),cudaMemcpyHostToDevice),"BuildGrid 16");
			hashMasksGpu[i].offsetTableSize = hashMasksCpu[i].offsetTableSize;
			hashMasksGpu[i].offsetTableMod = hashMasksCpu[i].offsetTableMod;
		}

		numBytes += NUMHASH*sizeof(HashMaskGpu);
		CCS(cudaMalloc((void**) &hashMask.dev_hashMaskGpu,NUMHASH*sizeof(HashMaskGpu)),"BuildGrid 15");
		CCS(cudaMemcpy(hashMask.dev_hashMaskGpu,hashMasksGpu,NUMHASH*sizeof(HashMaskGpu),cudaMemcpyHostToDevice),"BuildGrid 16");

		numBytes += keysSize*sizeof(t_octree_key);
		CCS(cudaMalloc((void**) &hashMask.dev_keys,keysSize*sizeof(t_octree_key)),"BuildGrid 15");
		CCS(cudaMemcpy(hashMask.dev_keys,keys,keysSize*sizeof(t_octree_key),cudaMemcpyHostToDevice),"BuildGrid 16");

		numBytes += weightsSize*sizeof(t_double);
		CCS(cudaMalloc((void**) &hashMask.dev_weights,weightsSize*sizeof(t_double)),"BuildGrid 15");
		CCS(cudaMemcpy(hashMask.dev_weights,weights,weightsSize*sizeof(t_double),cudaMemcpyHostToDevice),"BuildGrid 16");

#else

		numBytes += startIndsSize*sizeof(uint);
		CCS(cudaMalloc((void**) &elementOffsets.dev_startInds,startIndsSize*sizeof(uint)),"BuildGrid 5");
		CCS(cudaMemcpy(elementOffsets.dev_startInds,startInds,startIndsSize*sizeof(uint),cudaMemcpyHostToDevice),"BuildGrid 6");

		numBytes += offsetsLong1Size*sizeof(offsetD);
		CCS(cudaMalloc((void**) &elementOffsets.dev_offsets_long1,offsetsLong1Size*sizeof(offsetD)),"BuildGrid 7");
		CCS(cudaMemcpy(elementOffsets.dev_offsets_long1,offsetsLong1,offsetsLong1Size*sizeof(offsetD),cudaMemcpyHostToDevice),"BuildGrid 8");

		numBytes += offsetsLong2Size*sizeof(offsetD);
		CCS(cudaMalloc((void**) &elementOffsets.dev_offsets_long2,offsetsLong2Size*sizeof(offsetD)),"BuildGrid 9");
		CCS(cudaMemcpy(elementOffsets.dev_offsets_long2,offsetsLong2,offsetsLong2Size*sizeof(offsetD),cudaMemcpyHostToDevice),"BuildGrid 10");

		numBytes += offsetsLong3Size*sizeof(offsetD);
		CCS(cudaMalloc((void**) &elementOffsets.dev_offsets_long3,offsetsLong3Size*sizeof(offsetD)),"BuildGrid 11");
		CCS(cudaMemcpy(elementOffsets.dev_offsets_long3,offsetsLong3,offsetsLong3Size*sizeof(offsetD),cudaMemcpyHostToDevice),"BuildGrid 12");

		numBytes += offsetsLastSize*sizeof(uint);
		CCS(cudaMalloc((void**) &elementOffsets.dev_offsets_last,offsetsLastSize*sizeof(uint)),"BuildGrid 13");
		CCS(cudaMemcpy(elementOffsets.dev_offsets_last,offsetsLast,offsetsLastSize*sizeof(uint),cudaMemcpyHostToDevice),"BuildGrid 14");

		if (fineStartIndsSize != 0) {
			numBytes += fineStartIndsSize*sizeof(t_index);
			CCS(cudaMalloc((void**) &elementOffsets.dev_fineStartInds,fineStartIndsSize*sizeof(t_index)),"BuildGrid 1");
			CCS(cudaMemcpy(elementOffsets.dev_fineStartInds,fineStartInds,fineStartIndsSize*sizeof(t_index),cudaMemcpyHostToDevice),"BuildGrid 2");

			numBytes += localIndsSize*sizeof(char);
			CCS(cudaMalloc((void**) &elementOffsets.dev_localInds,localIndsSize*sizeof(uchar)),"BuildGrid 3");
			CCS(cudaMemcpy(elementOffsets.dev_localInds,localInds,localIndsSize*sizeof(uchar),cudaMemcpyHostToDevice),"BuildGrid 4");
		}

		numBytes += weightsSize*sizeof(t_double);
		CCS(cudaMalloc((void**) &elementOffsets.dev_weights,weightsSize*sizeof(t_double)),"BuildGrid 15");
		CCS(cudaMemcpy(elementOffsets.dev_weights,weights,weightsSize*sizeof(t_double),cudaMemcpyHostToDevice),"BuildGrid 16");

		delete startInds;
		delete offsetsLong1;
		delete offsetsLong2;
		delete offsetsLong3;
		delete offsetsLast;
		if (fineStartIndsSize != 0) {
			delete fineStartInds;
			delete localInds;
		}
#endif

		gpuMM->AllocateMemory(Grid,syncData.level,numBytes);

		if (syncData.switching) {
			CCS(cudaMalloc((void**) &syncData.gpuVecA,syncData.gpuVecSize*sizeof(t_double)),"Ddf");
			CCS(cudaMalloc((void**) &syncData.gpuVecB,syncData.gpuVecSize*sizeof(t_double)),"Ddf");
			gpuMM->AllocateMemory(DispVec,syncData.level,syncData.gpuVecSize*sizeof(t_double));
			gpuMM->AllocateMemory(DispVec,syncData.level,syncData.gpuVecSize*sizeof(t_double));
			syncData.gVec = syncData.gpuVecB;
		} else {

			numBytes = gpuBufferSize*sizeof(t_double);
			CCS(cudaMalloc((void**) &syncData.gpuBuffer,gpuBufferSize*sizeof(t_double)),"BuildGrid 17");

			numBytes += gpuIndsSize*sizeof(t_index);
			CCS(cudaMalloc((void**) &syncData.gpuInds,gpuIndsSize*sizeof(t_index)),"BuildGrid 18");
			CCS(cudaMemcpy(syncData.gpuInds,syncData.cpuInds,gpuIndsSize*sizeof(t_index),cudaMemcpyHostToDevice),"BuildGrid 19");

			gpuMM->AllocateMemory(Buffer,syncData.level,numBytes);

			delete syncData.cpuInds;

			syncData.gVec = 0;
		}
		delete weights;
	}

};

#endif /* GPUGRIDCONTAINER_H_ */
