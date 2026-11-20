/*
 * GPUGridBuilderOffsetSF.h
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

#ifndef GPUGRIDBUILDEROFFSETSF_H_
#define GPUGRIDBUILDEROFFSETSF_H_

#include <vector>
#include <mpi.h>
#include "Config.h"
#include <string.h>
#include "GPUGridBuilder.h"
#include "ApplyKernels.h"
#include "OctreeGrid.h"
#include "GPUGridContainer.h"
#include <omp.h>

template <class T>
class GPUGridBuilderOffsetSF : public GPUGridBuilder<T> {

public:

	GPUGridBuilderOffsetSF(GPUMemoryManager *gpuMM) : GPUGridBuilder<T>(gpuMM) {

	}

	virtual ~GPUGridBuilderOffsetSF() {

	}

	virtual GPUGridContainer BuildGrid(OctreeGrid<T> *oGrid,int level,MlOctreeGrid<T> *mlGrid = 0);

};

template <class T>
GPUGridContainer GPUGridBuilderOffsetSF<T>::BuildGrid(OctreeGrid<T> *oGrid,int level,MlOctreeGrid<T> *mlGrid) {

	const int maxNumThreads = 16;
	int numThreads;
#pragma omp parallel
	{
		if (omp_get_thread_num() == 0) {
			numThreads = omp_get_num_threads();
		}
	}

	GPUGridContainer gpuGridContainer;

	std::vector<std::pair<Offset8,t_octree_key> > offsets1[maxNumThreads];
	std::vector<std::pair<Offset16,t_octree_key> > offsets2[maxNumThreads];
	std::vector<std::pair<Offset32,t_octree_key> > offsets3[maxNumThreads];
	std::vector<std::pair<Offset32,t_octree_key> > offsetNodes[maxNumThreads];
	ElementOffsets elementOffsets;

	//For each element determine which kind it is an compute the offsets.
	std::vector<t_double> weights1[maxNumThreads];
	std::vector<t_double> weights2[maxNumThreads];
	std::vector<t_double> weights3[maxNumThreads];

	std::vector<t_index> fineStartInds1[maxNumThreads];
	std::vector<t_index> fineStartInds2[maxNumThreads];
	std::vector<t_index> fineStartInds3[maxNumThreads];
	std::vector<t_index> fineStartIndsN[maxNumThreads];

	std::vector<uchar> localInds[maxNumThreads];
	std::vector<uchar> localIndsNodes[maxNumThreads];

#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		std::vector<OctreeNode>::iterator fIter;
		if (mlGrid != 0) {
			fIter = mlGrid->_finegrid._OctreeGrid.begin();
		}
		bool endReached = false;
		std::vector<OctreeNode>::iterator iter = oGrid->_OctreeGrid.begin() + tid*oGrid->_OctreeGrid.size()/numThreads;
		std::vector<OctreeNode>::iterator endIter = oGrid->_OctreeGrid.begin() + (tid + 1)*oGrid->_OctreeGrid.size()/numThreads;
		if (tid == numThreads - 1) {
			endIter = oGrid->_OctreeGrid.end();
		}
		for (;iter != endIter;iter++) {
			if (iter >= oGrid->_GridIteratorEnd) {
				endReached = true;
				if (mlGrid == 0) {
					break;
				}
			}
			uchar children = 0;
			t_index fineStartIndex = NOFINEKEYS;
			if (mlGrid != 0) {
				while (fIter != mlGrid->_finegrid._GridIteratorEnd && fIter->key >> 3 < iter->key) {
					fIter++;
				}
				if (fIter != mlGrid->_finegrid._GridIteratorEnd && fIter->key >> 3 == iter->key) {
					std::vector<uchar> localKeys;
					localKeys.push_back(fIter->key & 7);
					while (fIter + 1 != mlGrid->_finegrid._GridIteratorEnd && (fIter + 1)->key >> 3 == iter->key) {
						fIter++;
						localKeys.push_back(fIter->key & 7);
					}
					fIter++;
					for (uint i = 0;i < localKeys.size();i++) {
						switch (localKeys[i]) {
							case 0:
								children = children | 1;
								break;
							case 1:
								children = children | 2;
								break;
							case 2:
								children = children | 4;
								break;
							case 3:
								children = children | 8;
								break;
							case 4:
								children = children | 16;
								break;
							case 5:
								children = children | 32;
								break;
							case 6:
								children = children | 64;
								break;
							case 7:
								children = children | 128;
								break;
						}
					}

					//find the index of the first child in the fine displacement vector
					t_octree_key fineKey = iter->key * 8;
					uchar lOffset = children;
					if (lOffset) {
						uchar off = 0;
						while (!(lOffset & 1)) {
							off++;
							lOffset = lOffset >> 1;
						}
						fineKey += off;
						fineStartIndex = mlGrid->_finegrid.GetIndexOfKey(fineKey);
					} else {
						fineStartIndex = NOFINEKEYS;
					}
				}
			}

			if (!endReached && iter->w >= 0) {
				t_index inds[8];
				oGrid->SearchIndexesThreadSafe(inds,iter);
				t_index sind = inds[0];
				bool b1 = false;
				bool b2 = false;
				t_index in = inds[6];
				t_index diff = in - sind;
				if (diff >= 256 && diff < 65536) {
					b1 = true;
				} else if (diff >= 65536) {
					b2 = true;
				}
				if (!b1 && !b2) {
					Offset8 o;
					o.offsetKeys.start = sind;
					o.offsetKeys.keys[0] = children;
					for (int k = 1;k < 8;k++) {
						diff = inds[k] - sind;
						o.offsetKeys.keys[k] = (uchar) diff;
					}
					offsets1[tid].push_back(std::pair<Offset8,t_octree_key>(o,iter->key));
					weights1[tid].push_back(iter->w);
					if (mlGrid != 0) {
						fineStartInds1[tid].push_back(fineStartIndex);
					}
				} else if (b1 && !b2) {
					Offset16 o;
					o.offsetKeys.start = sind;
					o.offsetKeys.keys[0] = children;
					for (int k = 1;k < 8;k++) {
						t_index diff = inds[k] - sind;
						o.offsetKeys.keys[k] = (ushort) diff;
					}
					offsets2[tid].push_back(std::pair<Offset16,t_octree_key>(o,iter->key));
					weights2[tid].push_back(iter->w);
					if (mlGrid != 0) {
						fineStartInds2[tid].push_back(fineStartIndex);
					}
				} else {
					Offset32 o;
					for (int k = 0;k < 8;k++) {
						o.offsetInt[k] = inds[k];
					}
					offsets3[tid].push_back(std::pair<Offset32,t_octree_key>(o,iter->key));
					weights3[tid].push_back(iter->w);
					localInds[tid].push_back(children);
					if (mlGrid != 0) {
						fineStartInds3[tid].push_back(fineStartIndex);
					}
				}
			} else {
				if (children != 0) {
					t_index inds[8];
					oGrid->SearchIndexesThreadSafe(inds,iter);
					Offset32 o;
					for (int k = 0;k < 8;k++) {
						o.offsetInt[k] = inds[k];
					}
					offsetNodes[tid].push_back(std::pair<Offset32,t_octree_key>(o,iter->key));
					localIndsNodes[tid].push_back(children);
					fineStartIndsN[tid].push_back(fineStartIndex);
				}
			}
		}
	}

	elementOffsets.dev_startInds = 0;
	elementOffsets.dev_offsets_long1 = 0;
	elementOffsets.dev_offsets_long2 = 0;
	elementOffsets.dev_offsets_long3 = 0;
	elementOffsets.dev_offsets_last = 0;
	elementOffsets.dev_weights = 0;
	elementOffsets.dev_fineStartInds = 0;
	elementOffsets.dev_localInds = 0;

	SetRpLookupTable();

	std::vector<uint> startInds;
	std::vector<offsetD> offsets_long1;
	std::vector<offsetD> offsets_long2;
	std::vector<offsetD> offsets_long3;
	std::vector<uint> offsets_last;

	for (int j = 0;j < numThreads;j++) {
		for (uint i = 0;i < offsets3[j].size();i++) {
			startInds.push_back(offsets3[j][i].first.offsetInt[0]);
			offsetD d;
			d.asuInt[0] = offsets3[j][i].first.offsetInt[1];
			d.asuInt[1] = offsets3[j][i].first.offsetInt[2];
			offsets_long1.push_back(d);
			d.asuInt[0] = offsets3[j][i].first.offsetInt[3];
			d.asuInt[1] = offsets3[j][i].first.offsetInt[4];
			offsets_long2.push_back(d);
			d.asuInt[0] = offsets3[j][i].first.offsetInt[5];
			d.asuInt[1] = offsets3[j][i].first.offsetInt[6];
			offsets_long3.push_back(d);
			offsets_last.push_back(offsets3[j][i].first.offsetInt[7]);
		}
	}

	int toAlign = (32 - (startInds.size() % 32)) % 32;
	for (int j = 0;j < numThreads;j++) {
		for (uint i = 0;i < offsets2[j].size();i++) {
			startInds.push_back(offsets2[j][i].first.offsetKeys.start);
			if (toAlign > 0) {
				offsetD d;
				d.asuInt[0] = offsets2[j][i].first.offsetKeys.start + offsets2[j][i].first.offsetKeys.keys[1];
				d.asuInt[1] = offsets2[j][i].first.offsetKeys.start + offsets2[j][i].first.offsetKeys.keys[2];
				offsets_long1.push_back(d);
				d.asuInt[0] = offsets2[j][i].first.offsetKeys.start + offsets2[j][i].first.offsetKeys.keys[3];
				d.asuInt[1] = offsets2[j][i].first.offsetKeys.start + offsets2[j][i].first.offsetKeys.keys[4];
				offsets_long2.push_back(d);
				d.asuInt[0] = offsets2[j][i].first.offsetKeys.start + offsets2[j][i].first.offsetKeys.keys[5];
				d.asuInt[1] = offsets2[j][i].first.offsetKeys.start + offsets2[j][i].first.offsetKeys.keys[6];
				offsets_long3.push_back(d);
				offsets_last.push_back(offsets2[j][i].first.offsetKeys.start + offsets2[j][i].first.offsetKeys.keys[7]);
				localInds[numThreads - 1].push_back((uchar) offsets2[j][i].first.offsetKeys.keys[0]);
				toAlign--;
			} else {
				offsetD d;
				d.asuInt[0] = offsets2[j][i].first.offsetInt[1];
				d.asuInt[1] = offsets2[j][i].first.offsetInt[2];
				offsets_long1.push_back(d);
				d.asuInt[0] = offsets2[j][i].first.offsetInt[3];
				d.asuInt[1] = offsets2[j][i].first.offsetInt[4];
				offsets_long2.push_back(d);
			}
		}
	}

	toAlign = (32 - (startInds.size() % 32)) % 32;
	for (int j = 0;j < numThreads;j++) {
		for (uint i = 0;i < offsets1[j].size();i++) {
			startInds.push_back(offsets1[j][i].first.offsetInt[0]);
			if (toAlign != 0) {
				Offset16 o16;
				for (int k = 0;k < 8;k++) {
					o16.offsetKeys.keys[k] = offsets1[j][i].first.offsetKeys.keys[k];
				}
				offsetD d;
				d.asuInt[0] = o16.offsetInt[1];
				d.asuInt[1] = o16.offsetInt[2];
				offsets_long1.push_back(d);
				d.asuInt[0] = o16.offsetInt[3];
				d.asuInt[1] = o16.offsetInt[4];
				offsets_long2.push_back(d);
				toAlign--;
			} else {
				offsetD d;
				d.asuInt[0] = offsets1[j][i].first.offsetInt[1];
				d.asuInt[1] = offsets1[j][i].first.offsetInt[2];
				offsets_long1.push_back(d);
			}
		}
	}

	elementOffsets.breakPoint0 = offsets_last.size();
	elementOffsets.breakPoint1 = offsets_long2.size();
	elementOffsets.breakPoint2 = startInds.size();

	for (int j = 0;j < numThreads;j++) {
		for (uint i = 0;i < offsetNodes[j].size();i++) {
			if (offsetNodes[j][i].second != -1) {
				startInds.push_back(offsetNodes[j][i].first.offsetInt[0]);
				offsetD d;
				d.asuInt[0] = offsetNodes[j][i].first.offsetInt[1];
				d.asuInt[1] = offsetNodes[j][i].first.offsetInt[2];
				offsets_long1.push_back(d);
				d.asuInt[0] = offsetNodes[j][i].first.offsetInt[3];
				d.asuInt[1] = offsetNodes[j][i].first.offsetInt[4];
				offsets_long2.push_back(d);
				d.asuInt[0] = offsetNodes[j][i].first.offsetInt[5];
				d.asuInt[1] = offsetNodes[j][i].first.offsetInt[6];
				offsets_long3.push_back(d);
				offsets_last.push_back(offsetNodes[j][i].first.offsetInt[7]);
			}
		}
	}

	int fsN = 0;
	if (mlGrid != 0) {
		int fStartsSize = 0;
		int lIndsSize = 0;
		for (int j = 0;j < numThreads;j++) {
			fStartsSize += fineStartInds1[j].size();
			fStartsSize += fineStartInds2[j].size();
			fStartsSize += fineStartInds3[j].size();
			fStartsSize += fineStartIndsN[j].size();
			fsN += fineStartIndsN[j].size();
			lIndsSize += localInds[j].size();
			lIndsSize += localIndsNodes[j].size();
		}

		gpuGridContainer.fineStartIndsSize = fStartsSize;
		gpuGridContainer.fineStartInds = new t_index[fStartsSize];
		int offset = 0;
		for (int j = 0;j < numThreads;j++) {
			memcpy(gpuGridContainer.fineStartInds + offset,fineStartInds3[j].data(),fineStartInds3[j].size()*sizeof(t_index));
			offset += fineStartInds3[j].size();
		}
		for (int j = 0;j < numThreads;j++) {
			memcpy(gpuGridContainer.fineStartInds + offset,fineStartInds2[j].data(),fineStartInds2[j].size()*sizeof(t_index));
			offset += fineStartInds2[j].size();
		}
		for (int j = 0;j < numThreads;j++) {
			memcpy(gpuGridContainer.fineStartInds + offset,fineStartInds1[j].data(),fineStartInds1[j].size()*sizeof(t_index));
			offset += fineStartInds1[j].size();
		}
		for (int j = 0;j < numThreads;j++) {
			memcpy(gpuGridContainer.fineStartInds + offset,fineStartIndsN[j].data(),fineStartIndsN[j].size()*sizeof(t_index));
			offset += fineStartIndsN[j].size();
		}

		gpuGridContainer.localIndsSize = lIndsSize;
		gpuGridContainer.localInds = new uchar[lIndsSize];

		offset = 0;
		for (int j = 0;j < numThreads;j++) {
			memcpy(gpuGridContainer.localInds + offset,localInds[j].data(),localInds[j].size()*sizeof(uchar));
			offset += localInds[j].size();
		}
		for (int j = 0;j < numThreads;j++) {
			memcpy(gpuGridContainer.localInds + offset,localIndsNodes[j].data(),localIndsNodes[j].size()*sizeof(uchar));
			offset += localIndsNodes[j].size();
		}

	} else {
		gpuGridContainer.fineStartIndsSize = 0;
		gpuGridContainer.localIndsSize = 0;
	}

	gpuGridContainer.startIndsSize = startInds.size();
	gpuGridContainer.startInds = new uint[startInds.size()];
	memcpy(gpuGridContainer.startInds,startInds.data(),startInds.size()*sizeof(uint));

	gpuGridContainer.offsetsLong1Size = offsets_long1.size();
	gpuGridContainer.offsetsLong1 = new offsetD[offsets_long1.size()];
	memcpy(gpuGridContainer.offsetsLong1,offsets_long1.data(),offsets_long1.size()*sizeof(offsetD));

	gpuGridContainer.offsetsLong2Size = offsets_long2.size();
	gpuGridContainer.offsetsLong2 = new offsetD[offsets_long2.size()];
	memcpy(gpuGridContainer.offsetsLong2,offsets_long2.data(),offsets_long2.size()*sizeof(offsetD));

	gpuGridContainer.offsetsLong3Size = offsets_long3.size();
	gpuGridContainer.offsetsLong3 = new offsetD[offsets_long3.size()];
	memcpy(gpuGridContainer.offsetsLong3,offsets_long3.data(),offsets_long3.size()*sizeof(offsetD));

	gpuGridContainer.offsetsLastSize = offsets_last.size();
	gpuGridContainer.offsetsLast = new uint[offsets_last.size()];
	memcpy(gpuGridContainer.offsetsLast,offsets_last.data(),offsets_last.size()*sizeof(uint));

	int wSize = 0;
	for (int j = 0;j < numThreads;j++) {
		wSize += weights1[j].size();
		wSize += weights2[j].size();
		wSize += weights3[j].size();
	}

	gpuGridContainer.weightsSize = wSize;
	gpuGridContainer.weights = new t_double[wSize];
	int offset = 0;
	for (int j = 0;j < numThreads;j++) {
		memcpy(gpuGridContainer.weights + offset,weights3[j].data(),weights3[j].size()*sizeof(t_double));
		offset += weights3[j].size();
	}
	for (int j = 0;j < numThreads;j++) {
		memcpy(gpuGridContainer.weights + offset,weights2[j].data(),weights2[j].size()*sizeof(t_double));
		offset += weights2[j].size();
	}
	for (int j = 0;j < numThreads;j++) {
		memcpy(gpuGridContainer.weights + offset,weights1[j].data(),weights1[j].size()*sizeof(t_double));
		offset += weights1[j].size();
	}

	//print some informations about the generated grid
	uint oSize1 = elementOffsets.breakPoint2 - elementOffsets.breakPoint1;
	uint oSize2 = elementOffsets.breakPoint1 - elementOffsets.breakPoint0;
	uint oSize3 = elementOffsets.breakPoint0;
	double ratiosIn[3];
	ratiosIn[0] = (double) oSize1/wSize;
	ratiosIn[1] = (double) oSize2/wSize;
	ratiosIn[2] = (double) oSize3/wSize;
	double ratiosMax[3] = {0.0,0.0,0.0};
	double ratiosAvg[3] = {0.0,0.0,0.0};
	MPI_Allreduce(ratiosIn,ratiosMax,3,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
	MPI_Allreduce(ratiosIn,ratiosAvg,3,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

	int mpiSize;
	MPI_Comm_size(MPI_COMM_WORLD,&mpiSize);
	for (int i = 0;i < 3;i++) {
		ratiosAvg[i] = ratiosAvg[i]/mpiSize;
	}

	gpuGridContainer.elementOffsets = elementOffsets;
	gpuGridContainer.applyStarts = wSize;
	gpuGridContainer.rpStarts = wSize + fsN;

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

#endif /* GPUGRIDBUILDEROFFSETSF_H_ */
