/*
 * GPUMemoryManager.cpp
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

#include "GPUMemoryManager.h"
#include <mpi.h>

void GPUMemoryManager::AllocateMemory(KMemory kMemory,int level,long numBytes) {
	_curMemoryUsage += numBytes;
	timeval now;
	gettimeofday(&now, NULL);
	_timeStamps.push_back(now);
	_changes.push_back(numBytes);
	if (_memoryMap.find(level) == _memoryMap.end()) {
		LMemory lMemory;
		lMemory.sizes[kMemory] = numBytes;
		if (kMemory == DispVec) {
			lMemory.dispVectorSize = numBytes;
		}
		_memoryMap[level] = lMemory;
	} else {
		if (kMemory == DispVec) {
			_memoryMap[level].sizes[kMemory] += numBytes;
			_memoryMap[level].dispVectorSize = numBytes;
		} else {
			_memoryMap[level].sizes[kMemory] = numBytes;
		}
	}
	if (_curMemoryUsage > _max) {
		_max = _curMemoryUsage;
	}
}

void GPUMemoryManager::FreeMemory(KMemory kMemory,int level) {
	timeval now;
	gettimeofday(&now, NULL);
	long numBytes;
	if (kMemory == DispVec) {
		numBytes = _memoryMap[level].dispVectorSize;
		_memoryMap[level].sizes[kMemory] -= numBytes;
	} else {
		numBytes = _memoryMap[level].sizes[kMemory];
		_memoryMap[level].sizes[kMemory] = 0;
	}
	_changes.push_back(numBytes*(-1));
	_curMemoryUsage -= numBytes;
	_timeStamps.push_back(now);
}

void GPUMemoryManager::PrintMax() {
	std::cout << "max GPU memory usage: " << _max << std::endl;
}

long GPUMemoryManager::GetMax() {
	return _max;
}

void GPUMemoryManager::PrintDistribution(int pid) {
	int s = _memoryMap.size();
	int l = 0;
	while (s > 0) {
		if (_memoryMap.find(l) != _memoryMap.end()) {
			long ratiosIn[8];
			ratiosIn[0] = _memoryMap[l].sizes[DispVec];
			ratiosIn[1] = _memoryMap[l].sizes[Diag];
			ratiosIn[2] = _memoryMap[l].sizes[DotVector];
			ratiosIn[3] = _memoryMap[l].sizes[Buffer];
			ratiosIn[4] = _memoryMap[l].sizes[BCond];
			ratiosIn[5] = _memoryMap[l].sizes[Grid];
			ratiosIn[6] = _memoryMap[l].sizes[StiffMat];
			ratiosIn[7] = _memoryMap[l].sizes[SDia];
			long ratiosMax[8] = {0,0,0,0,0,0,0,0};
			long ratiosAvg[8] = {0,0,0,0,0,0,0,0};
			MPI_Allreduce(ratiosIn,ratiosMax,8,MPI_LONG,MPI_MAX,MPI_COMM_WORLD);
			MPI_Allreduce(ratiosIn,ratiosAvg,8,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);

			int mpiSize;
			MPI_Comm_size(MPI_COMM_WORLD,&mpiSize);
			double ratiosAv[8];
			for (int i = 0;i < 8;i++) {
				ratiosAv[i] = (double) ratiosAvg[i]/mpiSize;
			}

			COUTPRECISION(4);
			PCOUT(pid,"level: " << l << "\n");
			PCOUT(pid,"DispVec: " << (long) ratiosAv[0]);
			PCOUT(pid," Diag: " << (long) ratiosAv[1]);
			PCOUT(pid," DotVector: " << (long) ratiosAv[2] << "\n");
			PCOUT(pid,"Buffer: " << (long) ratiosAv[3]);
			PCOUT(pid," BCond: " << (long) ratiosAv[4]);
			PCOUT(pid," Grid: " << (long) ratiosAv[5]);
			PCOUT(pid," StiffMat: " << (long) ratiosAv[6]);
			PCOUT(pid," SDia: " << (long) ratiosAv[7] << "\n");



			//DispVec = 0,Diag = 1,DotVector = 2,Buffer = 3,BCond = 4,Grid = 5,StiffMat = 6,SDia = 7
			/*std::cout << "level: " << l << "\n";
			std::cout << "DispVec: " << _memoryMap[l].sizes[DispVec];
			std::cout << " Diag: " << _memoryMap[l].sizes[Diag];
			std::cout << " DotVector: " << _memoryMap[l].sizes[DotVector] << "\n";
			std::cout << "Buffer: " << _memoryMap[l].sizes[Buffer];
			std::cout << " BCond: " << _memoryMap[l].sizes[BCond];
			std::cout << " Grid: " << _memoryMap[l].sizes[Grid];
			std::cout << " StiffMat: " << _memoryMap[l].sizes[StiffMat];
			std::cout << " SDia: " << _memoryMap[l].sizes[SDia] << "\n";*/

			COUTPRECISION(13);
			s--;
		}
		l++;
	}
}

void GPUMemoryManager::PrintEvents() {
	long curMem = 0;
	std::cout << "time; memory usage; memory change" << std::endl;
	for (uint i = 0;i < _changes.size();i++) {
		long change = _changes[i];
		curMem += change;
		double time = difftime(&_timeStamps[0],&_timeStamps[i]);
		std::cout << time << "; " << curMem << "; " << change << std::endl;
	}
}

double GPUMemoryManager::difftime(timeval *start, timeval *end) {
	double res =0;
	res = end->tv_sec - start->tv_sec;
	res += (((double) end->tv_usec) -  start->tv_usec) / 1e6;
	return res;
}
