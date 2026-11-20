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



#ifndef _BOUNDARY_CONDITON_H_
#define _BOUNDARY_CONDITON_H_
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "Config.h"
#include "GPUMemoryManager.h"


//! This class holds the boundary conditions.

/*! The BoundaryCondition holds the indexes and the values of the boundary nodes. 
*/
class BoundaryCondition {

	public:
		//!Constructor
		BoundaryCondition() {
			devFixedNodesInd = 0;
			devFixedNodes = 0;
			devLoadedNodesInd = 0;
			devLoadedNodes = 0;
		}
		//!Destructor
		~BoundaryCondition() {
			if (_switching) {
				delete storedDisp;
			} else {
				_gpuMM->FreeMemory(BCond,_level);
				CCS(cudaFree(devStoredDisp),"~BoundaryCondition 1");
				CCS(cudaFree(devFixedNodesInd),"~BoundaryCondition 1");
				CCS(cudaFree(devFixedNodes),"~BoundaryCondition 2");
				CCS(cudaFree(devLoadedNodesInd),"~BoundaryCondition 3");
				CCS(cudaFree(devLoadedNodes),"~BoundaryCondition 4");
			}
		}

		//!it holds the index of the BC to access it in the displacements array
		std::vector<t_index> FixedNodesInd;
		//!FixedNodes hold the value of the BC.
		std::vector<t_double> FixedNodes;
		std::vector<t_boundary_node> FixedKeys;

		//!it holds the index of the BC to access it in the displacements array
		std::vector<t_index> LoadedNodesInd;
		//!FixedNodes hold the value of the BC.
		std::vector<t_double> LoadedNodes;
		std::vector<t_boundary_node> LoadedKeys;

		t_index *devFixedNodesInd;
		t_double *devFixedNodes;
		t_index *devLoadedNodesInd;
		t_double *devLoadedNodes;

		t_double *devStoredDisp;
		t_double *storedDisp;

		int _level;
		bool _switching;
		GPUMemoryManager *_gpuMM;

		//! \brief Generate BC
		/*! It reads in the boundary nodes list in and converts it to the local data structure.  
		 */
		void GenerateBC(std::vector<t_boundary_node> &fixedKeyList,std::vector<t_boundary_node> &fixedIndexList,std::vector<t_boundary_node> &loadedKeyList,std::vector<t_boundary_node> &loadedIndexList) {
			std::vector<t_boundary_node>::iterator it;
			std::vector<t_boundary_node>::iterator keyIt = fixedKeyList.begin();
			for (it = fixedIndexList.begin();it != fixedIndexList.end();it++,keyIt++) {
				FixedNodesInd.push_back(it->first*3+it->second.dir);
				FixedNodes.push_back(it->second.disp);
				FixedKeys.push_back(*keyIt);
			}

			keyIt = loadedKeyList.begin();
			for (it = loadedIndexList.begin();it != loadedIndexList.end();it++,keyIt++) {
				LoadedNodesInd.push_back(it->first*3+it->second.dir);
				LoadedNodes.push_back(it->second.disp);
				LoadedKeys.push_back(*keyIt);
			}

			return;
		}

		void MoveDataToGpu(int level,GPUMemoryManager *gpuMM,bool switching) {
			_gpuMM = gpuMM;
			_level = level;
			_switching = switching;
			if (_switching) {
				storedDisp = new t_double[FixedNodesInd.size()];
			} else {
				_gpuMM->AllocateMemory(BCond,_level,GetNumBytes());

				CCS(cudaMalloc((void**) &devStoredDisp,FixedNodes.size()*sizeof(t_double)),"GenerateBC 1");

				CCS(cudaMalloc((void**) &devFixedNodesInd,FixedNodesInd.size()*sizeof(t_index)),"GenerateBC 1");
				CCS(cudaMalloc((void**) &devFixedNodes,FixedNodes.size()*sizeof(t_double)),"GenerateBC 2");
				CCS(cudaMalloc((void**) &devLoadedNodesInd,LoadedNodesInd.size()*sizeof(t_index)),"GenerateBC 3");
				CCS(cudaMalloc((void**) &devLoadedNodes,LoadedNodes.size()*sizeof(t_double)),"GenerateBC 4");

				CCS(cudaMemcpy(devFixedNodesInd,FixedNodesInd.data(),FixedNodesInd.size()*sizeof(t_index),cudaMemcpyHostToDevice),"GenerateBC 5");
				CCS(cudaMemcpy(devFixedNodes,FixedNodes.data(),FixedNodes.size()*sizeof(t_double),cudaMemcpyHostToDevice),"GenerateBC 6");
				CCS(cudaMemcpy(devLoadedNodesInd,LoadedNodesInd.data(),LoadedNodesInd.size()*sizeof(t_index),cudaMemcpyHostToDevice),"GenerateBC 7");
				CCS(cudaMemcpy(devLoadedNodes,LoadedNodes.data(),LoadedNodes.size()*sizeof(t_double),cudaMemcpyHostToDevice),"GenerateBC 8");
			}
		}
		
		long GetNumBytes() {
			long numBytes = FixedNodesInd.size()*sizeof(t_index);
			numBytes += 2*FixedNodes.size()*sizeof(t_double);
			numBytes += LoadedNodesInd.size()*sizeof(t_index);
			numBytes += LoadedNodes.size()*sizeof(t_double);
			return numBytes;
		}

		void UpdateBC(std::vector<t_boundary_node> &fixedIndexList,std::vector<t_boundary_node> &loadedIndexList,t_index centerSize) {
			std::vector< t_boundary_node >::iterator it;
			std::vector<t_boundary_node>::iterator keyIt = FixedKeys.begin();
			FixedNodesInd.resize(0);
			FixedNodes.resize(0);
			for (it = fixedIndexList.begin();it != fixedIndexList.end();it++,keyIt++) {
				FixedNodesInd.push_back(it->first*3+it->second.dir);
				FixedNodes.push_back(it->second.disp);
			}

			keyIt = LoadedKeys.begin();
			LoadedNodesInd.resize(0);
			LoadedNodes.resize(0);
			for (it = loadedIndexList.begin();it != loadedIndexList.end();it++,keyIt++) {
				LoadedNodesInd.push_back(it->first*3+it->second.dir);
				LoadedNodes.push_back(it->second.disp);
			}

			CCS(cudaMemcpy(devFixedNodesInd,FixedNodesInd.data(),FixedNodesInd.size()*sizeof(t_index),cudaMemcpyHostToDevice),"UpdateBC 1");
			CCS(cudaMemcpy(devFixedNodes,FixedNodes.data(),FixedNodes.size()*sizeof(t_double),cudaMemcpyHostToDevice),"UpdateBC 2");
			CCS(cudaMemcpy(devLoadedNodesInd,LoadedNodesInd.data(),LoadedNodesInd.size()*sizeof(t_index),cudaMemcpyHostToDevice),"UpdateBC 3");
			CCS(cudaMemcpy(devLoadedNodes,LoadedNodes.data(),LoadedNodes.size()*sizeof(t_double),cudaMemcpyHostToDevice),"UpdateBC 4");
		}


};

#endif
