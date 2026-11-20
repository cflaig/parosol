/*
 * GPUGridBuilderHashSF.h
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

#ifndef GPUGRIDBUILDERHASHSF_H_
#define GPUGRIDBUILDERHASHSF_H_

#include <vector>
#include <mpi.h>
#include <omp.h>
#include <string.h>
#include "Config.h"
#include "ApplyKernels.h"
#include "OctreeGrid.h"
#include "GPUGridContainer.h"

template <class T>
class GPUGridBuilderHashSF : public GPUGridBuilder<T> {

public:

	GPUGridBuilderHashSF(GPUMemoryManager *gpuMM) : GPUGridBuilder<T>(gpuMM) {

	}

	virtual ~GPUGridBuilderHashSF() {

	}

	virtual GPUGridContainer BuildGrid(OctreeGrid<T> *oGrid,int level,MlOctreeGrid<T> *mlGrid = 0);

};

template <class T>
GPUGridContainer GPUGridBuilderHashSF<T>::BuildGrid(OctreeGrid<T> *oGrid,int level,MlOctreeGrid<T> *mlGrid) {

	GPUGridContainer gpuGridContainer;

	PsHash *hashes[NUMHASH];

	int i;
#pragma omp parallel for
	for (i = 0;i < NUMHASH;i++) {
		hashes[i] = new PsHash(oGrid->_OctreeGrid,i,NUMHASH);
	}
	long totSizeHash = 0;
	long totSizeOffset = 0;

	//build the NUMHASH hashing functions
	for (int i = 0;i < NUMHASH;i++) {
		gpuGridContainer.hashMasksCpu[i].dev_hashTable = new t_index[hashes[i]->GetHashTableSize()];
		memcpy(gpuGridContainer.hashMasksCpu[i].dev_hashTable,hashes[i]->GetHashTable().data(),hashes[i]->GetHashTableSize()*sizeof(t_index));
		gpuGridContainer.hashMasksCpu[i].dev_offsetTable = new t_offset[hashes[i]->GetOffsetTableSize()];
		memcpy(gpuGridContainer.hashMasksCpu[i].dev_offsetTable,hashes[i]->GetOffsetTable().data(),hashes[i]->GetOffsetTableSize()*sizeof(t_offset));

		totSizeHash += hashes[i]->GetHashTableSize();
		totSizeOffset += hashes[i]->GetOffsetTableSize();

		gpuGridContainer.hashMasksCpu[i].hashTableSize = hashes[i]->GetHashTableSize();
		t_octree_key mod = 0X100000000;
		gpuGridContainer.hashMasksCpu[i].hashTableMod = (t_index) (mod % hashes[i]->GetHashTableSize());
		gpuGridContainer.hashMasksCpu[i].offsetTableSize = hashes[i]->GetOffsetTableSize();
		gpuGridContainer.hashMasksCpu[i].offsetTableMod = (t_index) (mod % hashes[i]->GetOffsetTableSize());

		delete hashes[i];
	}

	T KeyGen;
	gpuGridContainer.hashMask.maskx = KeyGen.maskx;
	gpuGridContainer.hashMask.masky = KeyGen.masky;
	gpuGridContainer.hashMask.maskz = KeyGen.maskz;

	//generate the key and weight arrays
	std::vector<t_octree_key> keys;
	std::vector<t_double> weights;
	std::vector<t_octree_key> nodes;

	for (std::vector<OctreeNode>::iterator iter = oGrid->_OctreeGrid.begin();iter != oGrid->_GridIteratorEnd;iter++) {
		if (iter->w >= 0) {
			keys.push_back(iter->key);
			weights.push_back(iter->w);
		} else {
			nodes.push_back(iter->key);
		}
	}

	keys.insert(keys.end(),nodes.begin(),nodes.end());

	gpuGridContainer.keysSize = keys.size();
	gpuGridContainer.keys = new t_octree_key[keys.size()];
	memcpy(gpuGridContainer.keys,keys.data(),keys.size()*sizeof(t_octree_key));
	gpuGridContainer.weightsSize = weights.size();
	gpuGridContainer.weights = new t_double[weights.size()];
	memcpy(gpuGridContainer.weights,weights.data(),weights.size()*sizeof(t_double));

	if (mlGrid != 0) {
		gpuGridContainer.hashMask.hashMaskFine = &mlGrid->_finegrid._gpuGridContainer.hashMask;
		gpuGridContainer.rpStarts = mlGrid->_finegrid.GetNrPrivateNodes();
	} else {
		gpuGridContainer.hashMask.hashMaskFine = 0;
		gpuGridContainer.rpStarts = 0;
	}

	//print some informations about the generated grid
	long ratiosIn[3];
	ratiosIn[0] = totSizeHash;
	ratiosIn[1] = totSizeOffset;
	long ratiosMax[2] = {0,0};
	long ratiosAvg[2] = {0,0};
	MPI_Allreduce(ratiosIn,ratiosMax,2,MPI_LONG,MPI_MAX,MPI_COMM_WORLD);
	MPI_Allreduce(ratiosIn,ratiosAvg,2,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);

	int mpiSize;
	MPI_Comm_size(MPI_COMM_WORLD,&mpiSize);
	double ratiosAv[2];
	for (int i = 0;i < 2;i++) {
		ratiosAv[i] = (double) ratiosAvg[i]/mpiSize;
	}

	gpuGridContainer.applyStarts = weights.size();

	gpuGridContainer.syncData.switching = false;
	gpuGridContainer.syncData.compressDia = false;
	gpuGridContainer.syncData.gVec = 0;
	gpuGridContainer.syncData.level = level;
	gpuGridContainer.syncData.gpuVecSize = oGrid->GetNrDofs();
	gpuGridContainer.syncData.dotStartSize = oGrid->GetNrPrivateNodes()*3;
	gpuGridContainer.syncData.bufferSize = oGrid->_remote_elements_index.size();
	gpuGridContainer.syncData.tailSize = oGrid->GetNrNodes() - oGrid->GetNrPrivateNodes();

	t_index bufferSize = gpuGridContainer.syncData.bufferSize*3;
	gpuGridContainer.gpuBufferSize = bufferSize;

	gpuGridContainer.gpuIndsSize = gpuGridContainer.syncData.bufferSize;
	gpuGridContainer.syncData.cpuInds = new t_index[gpuGridContainer.gpuIndsSize];
	memcpy(gpuGridContainer.syncData.cpuInds,oGrid->_remote_elements_index.data(),gpuGridContainer.gpuIndsSize*sizeof(t_index));

	gpuGridContainer.syncData.cpuBuffer = new t_double[(gpuGridContainer.syncData.bufferSize + gpuGridContainer.syncData.tailSize)*3];

	return gpuGridContainer;

}

#endif /* GPUGRIDBUILDERHASHSF_H_ */
