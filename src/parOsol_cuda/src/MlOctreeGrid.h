/*
 * GParOSol: a parallel FE solver for trabecular bone modeling
 * Copyright (C) 2011, Cyril Flaig
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

#ifndef MLOCTREEGRID_H
#define MLOCTREEGRID_H

#include <ostream>

#include <vector>

#include <utility>
#include <algorithm>

#include "OctreeGrid.h"
#include "ApplyKernels.h"
#include "OtherKernels.h"


/*! The MlOctreeGrid is base on the standard grid. It adds the restriction and the prolongation operations.
 * The corresponding key of the coarse grid is the fine grid key divided by 8.
 */
template <typename T>
class MlOctreeGrid : public OctreeGrid<T>
{
	public:
		MlOctreeGrid(OctreeGrid<T> *finegrid,GPUGridBuilder<T> *gpuGridBuilder,Timer *timer,int level):_finegrid(*finegrid) {
			OctreeGrid<T>::_level = level;
			OctreeGrid<T>::_My_PID = finegrid->_My_PID;
			OctreeGrid<T>::_Nr_CPU = finegrid->_Nr_CPU;
			OctreeGrid<T>::_gpuGridBuilder = gpuGridBuilder;
			OctreeGrid<T>::poisons_ratio = finegrid->poisons_ratio;
			Coarse();
			OctreeGrid<T>::_gpuGridContainer = OctreeGrid<T>::_gpuGridBuilder->BuildGrid(this,level,this);
		}

		virtual ~MlOctreeGrid()	{

		}

		virtual long GetNumBytes() {
			long numBytes = _finegrid.GetNumBytes();

			numBytes += OctreeGrid<T>::_gpuGridContainer.GetNumBytes();
			numBytes += OctreeGrid<T>::bc.GetNumBytes();
			numBytes += (int) ceil((double) OctreeGrid<T>::_gpuGridContainer.syncData.dotStartSize/256);
			numBytes += OctreeGrid<T>::_gpuGridContainer.syncData.gpuVecSize*sizeof(t_double);
			return numBytes;
		}

		virtual void ShiftDataToGPU(bool compressDia,bool turnOnSwitching) {
			if (compressDia) {
				OctreeGrid<T>::_gpuGridContainer.syncData.compressDia = true;
			}
			_finegrid.ShiftDataToGPU(compressDia,turnOnSwitching);

			OctreeGrid<T>::_gpuGridContainer.MoveDataToGpu(OctreeGrid<T>::_gpuGridBuilder->_gpuMM);
			OctreeGrid<T>::bc.MoveDataToGpu(OctreeGrid<T>::_gpuGridContainer.syncData.level,OctreeGrid<T>::_gpuGridBuilder->_gpuMM,false);

			OctreeGrid<T>::_gpuGrid = new GPUGrid(OctreeGrid<T>::_gpuGridContainer,OctreeGrid<T>::_gpuGridBuilder->_gpuMM);
			OctreeGrid<T>::InitializeDotVector();
			if (_finegrid._gpuGrid->_syncData.gVec != 0) {
				OctreeGrid<T>::_gpuGrid->_syncData.gVec = _finegrid._gpuGrid->_syncData.gVec;
			}
		}

		virtual SyncData* GetFinestSyncData() {
			return _finegrid.GetFinestSyncData();
		}

		virtual bool IsFinestGrid() {
			return false;
		}


		void Restrict(DispVector &fine,DispVector &coarse,cudaStream_t streamFine = 0,cudaStream_t streamCoarse = 0);
		void Prolongate(DispVector &coarse,DispVector &fine,cudaStream_t streamCoarse = 0,cudaStream_t streamFine = 0);

		OctreeGrid<T> &_finegrid;

	private:
		void Coarse();

		void coarse_member() {
			t_coord dims[3];
			_finegrid.GetGlobalDim(dims);
			for(int i =0; i < 3; i++)
				OctreeGrid<T>::gdim[i] = (dims[i]+1)/2;

			_finegrid.GetLocalDim(dims);
			for(int i =0; i < 3; i++)
				OctreeGrid<T>::ldim[i] = (dims[i]+1)/2;

			double res[3];
			_finegrid.GetRes(res);
			for(int i =0; i < 3; i++)
				OctreeGrid<T>::res[i] = res[i]*2;

			OctreeGrid<T>::_smallest_elements.resize(OctreeGrid<T>::_Nr_CPU+1);
			for(int i =0; i < OctreeGrid<T>::_Nr_CPU+1; i++)
				OctreeGrid<T>::_smallest_elements[i] = _finegrid._smallest_elements[i]/8;
			OctreeGrid<T>::_smallest_elements[OctreeGrid<T>::_Nr_CPU] +=8; //round up

		}

		int SetBoundaryNodes(BoundaryCondition &bc,DispVector &x,cudaStream_t stream = 0) {
			SetBoundaries2K(bc,x,stream);
			return 0;
		}
	
	void Distribute_Elements(std::set<OctreeNode> &octset) {
		std::set<OctreeNode>::iterator it;
		std::pair<std::set<OctreeNode>::iterator, bool> ret;
		OctreeNode tmp_node;

		//copy things in to an vector
		it = octset.lower_bound(OctreeNode(OctreeGrid<T>::_smallest_elements[OctreeGrid<T>::_My_PID+1],0));
		OctreeGrid<T>::_OctreeGrid.resize(0);
		OctreeGrid<T>::_OctreeGrid.reserve(octset.size());
		OctreeGrid<T>::_OctreeGrid.insert(OctreeGrid<T>::_OctreeGrid.begin(), it, octset.end());

		std::vector<OctreeNode> offcpu_elem;

		std::vector<OctreeNode>::iterator Oct_iter = OctreeGrid<T>::_OctreeGrid.begin();
		std::vector< std::vector<OctreeNode>::iterator> send_pos_iters(OctreeGrid<T>::_Nr_CPU+1);
		std::vector<int> send_count(OctreeGrid<T>::_Nr_CPU);
		std::vector<int> send_pos(OctreeGrid<T>::_Nr_CPU);
		std::vector<int> recv_count(OctreeGrid<T>::_Nr_CPU);
		std::vector<int> recv_pos(OctreeGrid<T>::_Nr_CPU);

		send_pos_iters[0] = OctreeGrid<T>::_OctreeGrid.begin();
		send_pos_iters[OctreeGrid<T>::_Nr_CPU] = OctreeGrid<T>::_OctreeGrid.end();

		//Prepare Position which belongs to other CPU
		for(int i = 0; i < OctreeGrid<T>::_Nr_CPU; i++)
		{
			while( (Oct_iter != OctreeGrid<T>::_OctreeGrid.end()) && (Oct_iter->key < OctreeGrid<T>::_smallest_elements[i+1]))
				Oct_iter++;
			send_pos_iters[i+1] = Oct_iter;
		}

		//Compute how much date to send
		for(int i = 0; i < OctreeGrid<T>::_Nr_CPU; i++) {
			send_count[i] = send_pos_iters[i+1] - send_pos_iters[i];
		}
		for(int i = 0; i < OctreeGrid<T>::_Nr_CPU; i++) {
			send_count[i] *= sizeof(OctreeNode);
		}

		MPI_Alltoall (&send_count[0], 1, MPI_INT, &recv_count[0], 1, MPI_INT, MPI_COMM_WORLD);

		//Compute the Positions
		send_pos[0] = 0;
		recv_pos[0] = 0;
		for(int i=1; i < OctreeGrid<T>::_Nr_CPU; i++) {
			send_pos[i] = (send_pos[i-1] + send_count[i-1]);
			recv_pos[i] = recv_pos[i-1] + recv_count[i-1];
		}

		int newsize = (recv_pos[OctreeGrid<T>::_Nr_CPU-1] + recv_count[OctreeGrid<T>::_Nr_CPU-1])/sizeof(OctreeNode);
		offcpu_elem.resize(newsize);

		void * send_addr;
		void * recv_addr;
		if (OctreeGrid<T>::_OctreeGrid.size() > 0)
			send_addr = &OctreeGrid<T>::_OctreeGrid[0];
		else
			send_addr = NULL;

		if (offcpu_elem.size() > 0)
			recv_addr = &offcpu_elem[0];
		else
			recv_addr = NULL;

		MPI_Alltoallv(send_addr, &send_count[0], &send_pos[0], MPI_CHAR,
				recv_addr, &recv_count[0], &recv_pos[0], MPI_CHAR,
				MPI_COMM_WORLD);

		//include int to the set

		for(t_index i =0; i< offcpu_elem.size(); i++) {
			tmp_node = offcpu_elem[i];
			ret = octset.insert(tmp_node);
			if (ret.second) {   //inserted
				if(tmp_node.w > 0) {
					OctreeGrid<T>::_nr_elem++;
				}
			} else {     // already in the set update the weight when it is an element
				if (tmp_node.w >0) {
					if (ret.first->w >0) {   // add up only if an element is in the set
						tmp_node.w += ret.first->w;
					} 
					octset.erase(ret.first);
					ret = octset.insert(tmp_node);
				}
			}
		}
		OctreeGrid<T>::_OctreeGrid.resize(0);
		return;
	}

};

template <class T>
void MlOctreeGrid<T>::Coarse() {

	const int maxNumThreads = 16;
	int numThreads;
#pragma omp parallel
	{
		if (omp_get_thread_num() == 0) {
			numThreads = omp_get_num_threads();
		}
	}

	coarse_member();
	t_octree_iterator iter;

	std::pair<std::set<OctreeNode>::iterator, bool> ret;
	std::set<OctreeNode>::iterator it;
	std::set<OctreeNode>::iterator it_inserted;
	std::set<OctreeNode> octset;
	OctreeNode tmp_node(0,0);
	t_octree_key coarsekey;
	//t_octree_key nodekeys[8];
	double weight;

	OctreeGrid<T>::_nr_elem =0;
	OctreeGrid<T>::_OctreeGrid.resize(0);
	//coarse the grid
	for(iter = _finegrid._OctreeGrid.begin();iter != _finegrid._GridIteratorEnd;iter++) {
		weight = 0;
		coarsekey = iter->key >> 3;
		if (iter->w > 0) {
			weight += iter->w/8;
		}
		while (iter + 1 != _finegrid._GridIteratorEnd && (iter + 1)->key >> 3 == coarsekey) {
			iter++;
			if (iter->w > 0) {
				weight += iter->w/8;
			}
		}
		if (weight > 0) {
			OctreeGrid<T>::_nr_elem++;
			OctreeGrid<T>::_OctreeGrid.push_back(OctreeNode(coarsekey,weight));
		}
	}

	std::set<OctreeNode> tNodes[maxNumThreads];

#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		t_octree_iterator oIt = OctreeGrid<T>::_OctreeGrid.begin() + tid*OctreeGrid<T>::_OctreeGrid.size()/numThreads;
		t_octree_iterator endIt = OctreeGrid<T>::_OctreeGrid.begin() + (tid + 1)*OctreeGrid<T>::_OctreeGrid.size()/numThreads;
		if (tid == numThreads - 1) {
			endIt = OctreeGrid<T>::_OctreeGrid.end();
		}
		t_octree_key keys[8];
		for (;oIt != endIt;oIt++) {
			OctreeGrid<T>::ComputeKeysofNodesZCurve(keys,oIt->key);
			for (int i = 1;i < 8;i++) {
				if (OctreeGrid<T>::GetIndexOfKey(keys[i]) == -1) {
					tNodes[tid].insert(OctreeNode(keys[i],-2.0));
				}
			}
		}
	}

	std::vector<OctreeNode> finalVec;
	if (numThreads == 1) {
		finalVec.resize(OctreeGrid<T>::_OctreeGrid.size() + tNodes[0].size());
		std::merge(OctreeGrid<T>::_OctreeGrid.begin(),OctreeGrid<T>::_OctreeGrid.end(),tNodes[0].begin(),tNodes[0].end(),finalVec.begin());
	} else if (numThreads == 2) {
		std::vector<OctreeNode> tVector(tNodes[0].size() + tNodes[1].size());
		t_octree_iterator uIter = std::set_union(tNodes[0].begin(),tNodes[0].end(),tNodes[1].begin(),tNodes[1].end(),tVector.begin());
		tVector.resize(uIter - tVector.begin());
		finalVec.resize(OctreeGrid<T>::_OctreeGrid.size() + tVector.size());
		std::merge(OctreeGrid<T>::_OctreeGrid.begin(),OctreeGrid<T>::_OctreeGrid.end(),tVector.begin(),tVector.end(),finalVec.begin());
	} else if (numThreads == 4) {
		std::vector<OctreeNode> tVector1(tNodes[0].size() + tNodes[1].size());
		std::vector<OctreeNode> tVector2(tNodes[2].size() + tNodes[3].size());
#pragma omp parallel num_threads(2)
		{
			int tid = omp_get_thread_num();
			if (tid == 0) {
				t_octree_iterator uIter = std::set_union(tNodes[0].begin(),tNodes[0].end(),tNodes[1].begin(),tNodes[1].end(),tVector1.begin());
				tVector1.resize(uIter - tVector1.begin());
			} else {
				t_octree_iterator uIter = std::set_union(tNodes[2].begin(),tNodes[2].end(),tNodes[3].begin(),tNodes[3].end(),tVector2.begin());
				tVector2.resize(uIter - tVector2.begin());
			}
		}
		std::vector<OctreeNode> tVector3(tVector1.size() + tVector2.size());
		t_octree_iterator uIter = std::set_union(tVector1.begin(),tVector1.end(),tVector2.begin(),tVector2.end(),tVector3.begin());
		tVector3.resize(uIter - tVector3.begin());
		finalVec.resize(OctreeGrid<T>::_OctreeGrid.size() + tVector3.size());
		std::merge(OctreeGrid<T>::_OctreeGrid.begin(),OctreeGrid<T>::_OctreeGrid.end(),tVector3.begin(),tVector3.end(),finalVec.begin());
	} else if (numThreads == 8) {
		std::vector<OctreeNode> tVector1(tNodes[0].size() + tNodes[1].size());
		std::vector<OctreeNode> tVector2(tNodes[2].size() + tNodes[3].size());
		std::vector<OctreeNode> tVector3(tNodes[4].size() + tNodes[5].size());
		std::vector<OctreeNode> tVector4(tNodes[6].size() + tNodes[7].size());
#pragma omp parallel num_threads(4)
		{
			int tid = omp_get_thread_num();
			if (tid == 0) {
				t_octree_iterator uIter = std::set_union(tNodes[0].begin(),tNodes[0].end(),tNodes[1].begin(),tNodes[1].end(),tVector1.begin());
				tVector1.resize(uIter - tVector1.begin());
			} else if (tid == 1) {
				t_octree_iterator uIter = std::set_union(tNodes[2].begin(),tNodes[2].end(),tNodes[3].begin(),tNodes[3].end(),tVector2.begin());
				tVector2.resize(uIter - tVector2.begin());
			} else if (tid == 2) {
				t_octree_iterator uIter = std::set_union(tNodes[4].begin(),tNodes[4].end(),tNodes[5].begin(),tNodes[5].end(),tVector3.begin());
				tVector3.resize(uIter - tVector3.begin());
			} else {
				t_octree_iterator uIter = std::set_union(tNodes[6].begin(),tNodes[6].end(),tNodes[7].begin(),tNodes[7].end(),tVector4.begin());
				tVector4.resize(uIter - tVector4.begin());
			}
		}
		std::vector<OctreeNode> tVector5(tVector1.size() + tVector2.size());
		std::vector<OctreeNode> tVector6(tVector3.size() + tVector4.size());
#pragma omp parallel num_threads(2)
		{
			int tid = omp_get_thread_num();
			if (tid == 0) {
				t_octree_iterator uIter = std::set_union(tVector1.begin(),tVector1.end(),tVector2.begin(),tVector2.end(),tVector5.begin());
				tVector5.resize(uIter - tVector5.begin());
			} else {
				t_octree_iterator uIter = std::set_union(tVector3.begin(),tVector3.end(),tVector4.begin(),tVector4.end(),tVector6.begin());
				tVector6.resize(uIter - tVector6.begin());
			}
		}
		std::vector<OctreeNode> tVector7(tVector5.size() + tVector6.size());
		t_octree_iterator uIter = std::set_union(tVector5.begin(),tVector5.end(),tVector6.begin(),tVector6.end(),tVector7.begin());
		tVector7.resize(uIter - tVector7.begin());
		finalVec.resize(OctreeGrid<T>::_OctreeGrid.size() + tVector7.size());
		std::merge(OctreeGrid<T>::_OctreeGrid.begin(),OctreeGrid<T>::_OctreeGrid.end(),tVector7.begin(),tVector7.end(),finalVec.begin());
	} else if (numThreads == 16) {
		std::vector<OctreeNode> tVector1(tNodes[0].size() + tNodes[1].size());
		std::vector<OctreeNode> tVector2(tNodes[2].size() + tNodes[3].size());
		std::vector<OctreeNode> tVector3(tNodes[4].size() + tNodes[5].size());
		std::vector<OctreeNode> tVector4(tNodes[6].size() + tNodes[7].size());
		std::vector<OctreeNode> tVector5(tNodes[8].size() + tNodes[9].size());
		std::vector<OctreeNode> tVector6(tNodes[10].size() + tNodes[11].size());
		std::vector<OctreeNode> tVector7(tNodes[12].size() + tNodes[13].size());
		std::vector<OctreeNode> tVector8(tNodes[14].size() + tNodes[15].size());
#pragma omp parallel num_threads(8)
		{
			int tid = omp_get_thread_num();
			if (tid == 0) {
				t_octree_iterator uIter = std::set_union(tNodes[0].begin(),tNodes[0].end(),tNodes[1].begin(),tNodes[1].end(),tVector1.begin());
				tVector1.resize(uIter - tVector1.begin());
			} else if (tid == 1) {
				t_octree_iterator uIter = std::set_union(tNodes[2].begin(),tNodes[2].end(),tNodes[3].begin(),tNodes[3].end(),tVector2.begin());
				tVector2.resize(uIter - tVector2.begin());
			} else if (tid == 2) {
				t_octree_iterator uIter = std::set_union(tNodes[4].begin(),tNodes[4].end(),tNodes[5].begin(),tNodes[5].end(),tVector3.begin());
				tVector3.resize(uIter - tVector3.begin());
			} else if (tid == 3) {
				t_octree_iterator uIter = std::set_union(tNodes[6].begin(),tNodes[6].end(),tNodes[7].begin(),tNodes[7].end(),tVector4.begin());
				tVector4.resize(uIter - tVector4.begin());
			} else if (tid == 4) {
				t_octree_iterator uIter = std::set_union(tNodes[8].begin(),tNodes[8].end(),tNodes[9].begin(),tNodes[9].end(),tVector5.begin());
				tVector5.resize(uIter - tVector5.begin());
			} else if (tid == 5) {
				t_octree_iterator uIter = std::set_union(tNodes[10].begin(),tNodes[10].end(),tNodes[11].begin(),tNodes[11].end(),tVector6.begin());
				tVector6.resize(uIter - tVector6.begin());
			} else if (tid == 6) {
				t_octree_iterator uIter = std::set_union(tNodes[12].begin(),tNodes[12].end(),tNodes[13].begin(),tNodes[13].end(),tVector7.begin());
				tVector7.resize(uIter - tVector7.begin());
			} else {
				t_octree_iterator uIter = std::set_union(tNodes[14].begin(),tNodes[14].end(),tNodes[15].begin(),tNodes[15].end(),tVector8.begin());
				tVector8.resize(uIter - tVector8.begin());
			}
		}
		std::vector<OctreeNode> tVector9(tVector1.size() + tVector2.size());
		std::vector<OctreeNode> tVector10(tVector3.size() + tVector4.size());
		std::vector<OctreeNode> tVector11(tVector5.size() + tVector6.size());
		std::vector<OctreeNode> tVector12(tVector7.size() + tVector8.size());
#pragma omp parallel num_threads(4)
		{
			int tid = omp_get_thread_num();
			if (tid == 0) {
				t_octree_iterator uIter = std::set_union(tVector1.begin(),tVector1.end(),tVector2.begin(),tVector2.end(),tVector9.begin());
				tVector9.resize(uIter - tVector9.begin());
			} else if (tid == 1) {
				t_octree_iterator uIter = std::set_union(tVector3.begin(),tVector3.end(),tVector4.begin(),tVector4.end(),tVector10.begin());
				tVector10.resize(uIter - tVector10.begin());
			} else if (tid == 2) {
				t_octree_iterator uIter = std::set_union(tVector5.begin(),tVector5.end(),tVector6.begin(),tVector6.end(),tVector11.begin());
				tVector11.resize(uIter - tVector11.begin());
			} else {
				t_octree_iterator uIter = std::set_union(tVector7.begin(),tVector7.end(),tVector8.begin(),tVector8.end(),tVector12.begin());
				tVector12.resize(uIter - tVector12.begin());
			}
		}
		std::vector<OctreeNode> tVector13(tVector9.size() + tVector10.size());
		std::vector<OctreeNode> tVector14(tVector11.size() + tVector12.size());
#pragma omp parallel num_threads(2)
		{
			int tid = omp_get_thread_num();
			if (tid == 0) {
				t_octree_iterator uIter = std::set_union(tVector9.begin(),tVector9.end(),tVector10.begin(),tVector10.end(),tVector13.begin());
				tVector13.resize(uIter - tVector13.begin());
			} else {
				t_octree_iterator uIter = std::set_union(tVector11.begin(),tVector11.end(),tVector12.begin(),tVector12.end(),tVector14.begin());
				tVector14.resize(uIter - tVector14.begin());
			}
		}
		std::vector<OctreeNode> tVector15(tVector13.size() + tVector14.size());
		t_octree_iterator uIter = std::set_union(tVector13.begin(),tVector13.end(),tVector14.begin(),tVector14.end(),tVector15.begin());
		tVector15.resize(uIter - tVector15.begin());
		finalVec.resize(OctreeGrid<T>::_OctreeGrid.size() + tVector15.size());
		std::merge(OctreeGrid<T>::_OctreeGrid.begin(),OctreeGrid<T>::_OctreeGrid.end(),tVector15.begin(),tVector15.end(),finalVec.begin());
	}

	octset.clear();
	octset.insert(finalVec.begin(),finalVec.end());

	t_octree_key min = (((t_octree_key) 1)<< 50);
	t_octree_key max = -1;
	if (octset.size() > 0) {
		min = octset.begin()->key;
		max = octset.rbegin()->key;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	OctreeGrid<T>::Compute_min_max(min, max);

	Distribute_Elements(octset);

// adding missing nodes from prolongation


	//needed nodes in the coarse grid given the local fine coordinate
	//(except the first (parent) which is guaranteed to be in the set)
	uchar table[8];
	table[0] = 0;		// 00000000
	table[1] = 2;		// 00000010
	table[2] = 4;		// 00000100
	table[3] = 14;		// 00001110
	table[4] = 16;		// 00010000
	table[5] = 50;		// 00110010
	table[6] = 84;		// 01010100
	table[7] = 254;		// 11111110

	OctreeGrid<T>::_OctreeGrid.reserve(octset.size());
	OctreeGrid<T>::_OctreeGrid.resize(0);
	OctreeGrid<T>::_OctreeGrid.insert(OctreeGrid<T>::_OctreeGrid.begin(), octset.begin(), octset.end());

	std::set<OctreeNode> tempNodes[maxNumThreads];

#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		t_octree_iterator oIt = _finegrid._OctreeGrid.begin() + tid*(_finegrid._GridIteratorEnd - _finegrid._OctreeGrid.begin())/numThreads;
		t_octree_iterator endIt = _finegrid._OctreeGrid.begin() + (tid + 1)*(_finegrid._GridIteratorEnd - _finegrid._OctreeGrid.begin())/numThreads;
		if (tid == numThreads - 1) {
			endIt = _finegrid._GridIteratorEnd;
		}
		t_octree_key keys[8];
		t_octree_key coarseKey;
		for (;oIt != endIt;oIt++) {
			coarseKey = oIt->key >> 3;
			uchar localFineCoordinate = (oIt->key) & 7;
			uchar children = 0;
			children = children | table[localFineCoordinate];

			while (oIt + 1 != endIt/*_finegrid._GridIteratorEnd*/ && (oIt + 1)->key >> 3 == coarseKey) {
				oIt++;
				localFineCoordinate = (oIt->key) & 7;
				children = children | table[localFineCoordinate];
			}
			OctreeGrid<T>::ComputeKeysofNodesZCurve(keys,coarseKey);
			children = children >> 1;
			for (int i = 1;i < 8;i++) {
				if (children & 1) {
					OctreeNode tNode = OctreeNode(keys[i],-1.1111);
					int ind = OctreeGrid<T>::GetIndexOfKey(keys[i]);
					//t_octree_iterator found = octset.find(tNode);
					if (ind == -1) {
						tempNodes[tid].insert(tNode);
					}
				}
				children = children >> 1;
			}
		}
	}
	for (int i = 0;i < numThreads;i++) {
		octset.insert(tempNodes[i].begin(),tempNodes[i].end());
	}

	OctreeGrid<T>::_OctreeGrid.reserve(octset.size());
	OctreeGrid<T>::_OctreeGrid.resize(0);
	OctreeGrid<T>::_OctreeGrid.insert(OctreeGrid<T>::_OctreeGrid.begin(), octset.begin(), octset.end());

    OctreeGrid<T>::_nr_nodes = OctreeGrid<T>::_OctreeGrid.size();

	OctreeGrid<T>::Prepare_Communication();

	std::vector<t_boundary_node>::iterator bcIt;
	std::vector<t_index>::iterator dirIt;
	t_octree_key finekey;
	typedef t_boundary_node bcpair;
	std::set<bcpair> bcset;
	tmp_node.w = -1; // Does it work

	short LocalFineCoordinate;
	short max_x, max_y, max_z;
	t_octree_key new_bckey;
	boundary_disp bc_value;
	short direction;

	for (bcIt = _finegrid.bc.FixedKeys.begin();bcIt != _finegrid.bc.FixedKeys.end();bcIt++) {
		finekey  = bcIt->first;
		direction = bcIt->second.dir;
		
		tmp_node.key = finekey >> 3;
		LocalFineCoordinate = finekey & 7;

		max_x = (LocalFineCoordinate & 1);
		max_y = (LocalFineCoordinate & 2);
		max_z = (LocalFineCoordinate & 4);

		for (short z = 0;z <= max_z;z += 4) {
			for (short y = 0;y <= max_y;y += 2) {
				for (short x = 0;x <= max_x;x++) {
					new_bckey = finekey >> 3;
					if (z != 0) {
						new_bckey = OctreeGrid<T>::KeyGen.incZ(new_bckey);
					}
					if (y != 0) {
						new_bckey = OctreeGrid<T>::KeyGen.incY(new_bckey);
					}
					if (x != 0) {
						new_bckey = OctreeGrid<T>::KeyGen.incX(new_bckey);
					}

					//TODO only for z
					tmp_node.key = new_bckey;
				   	
					bcset.insert(bcpair(tmp_node, boundary_disp(direction,0)));
				}
			}
		}
	}

	OctreeGrid<T>::_BC_fixednode_list.reserve(bcset.size());
	OctreeGrid<T>::_BC_fixednode_list.insert(OctreeGrid<T>::_BC_fixednode_list.begin(), bcset.begin(), bcset.end());
	OctreeGrid<T>::init_Members();

}

template <class T>
void MlOctreeGrid<T>::Restrict(DispVector &fine,DispVector &coarse,cudaStream_t streamFine,cudaStream_t streamCoarse) {
	
	fine.PrepareIn(0);

	coarse.PrepareOut(0);

	_finegrid.Recv_import_Ghost(fine._syncData);
	_finegrid.Send_import_Ghost(fine._syncData);
	_finegrid.Wait_import_Ghost();

	fine.CopyTailToGpu();

	OctreeGrid<T>::_gpuGrid->RestrictK(coarse.GetVector(),fine.GetVector());

	coarse.CopyTailToCpu();

	OctreeGrid<T>::Recv_export_Ghost(coarse._syncData);
	OctreeGrid<T>::Send_export_Ghost(coarse._syncData);
	OctreeGrid<T>::WaitAndCopy_export_Ghost();

	coarse.PostProcess(0);

	SetBoundaryNodes(OctreeGrid<T>::bc,coarse);

	return;
}

template <class T>
void MlOctreeGrid<T>::Prolongate(DispVector &coarse,DispVector &fine,cudaStream_t streamCoarse,cudaStream_t streamFine) {

	SetBoundaryNodes(OctreeGrid<T>::bc,coarse);

	coarse.PrepareIn(0);
	fine.PrepareOut(0,true);

	OctreeGrid<T>::Recv_import_Ghost(coarse._syncData);
	OctreeGrid<T>::Send_import_Ghost(coarse._syncData);
	OctreeGrid<T>::Wait_import_Ghost();

	//coarse.CopyBufferToGpu();
	coarse.CopyTailToGpu();

	OctreeGrid<T>::_gpuGrid->ProlongateK(coarse.GetVector(),fine.GetVector(),streamFine);

	fine.CopyTailToCpu();
	SetBoundaryNodes(_finegrid.bc,fine);

}
#endif 

