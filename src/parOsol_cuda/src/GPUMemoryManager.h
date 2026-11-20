/*
 * GPUMemoryManager.h
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

#ifndef GPUMEMORYMANAGER_H_
#define GPUMEMORYMANAGER_H_

#include <vector>
#include <map>
#include <sys/time.h>
#include <string>
#include <iostream>
#include "Config.h"

enum KMemory {DispVec = 0,Diag = 1,DotVector = 2,Buffer = 3,BCond = 4,Grid = 5,StiffMat = 6,SDia = 7};

class GPUMemoryManager {

	public:

		struct LMemory {
			long sizes[8];
			long dispVectorSize;
			LMemory() {
				for (uint i = 0;i < 8;i++) {
					sizes[i] = 0;
				}
				dispVectorSize = 0;
			}
		};

		GPUMemoryManager() {
			_curMemoryUsage = 0;
			_max = 0;
		}

		void AllocateMemory(KMemory kMemory,int level,long numBytes);

		void FreeMemory(KMemory kMemory,int level);

		void PrintMax();

		long GetMax();

		void PrintDistribution(int pid);

		void PrintEvents();

	private:

		double difftime(timeval *start, timeval *end);

		long _curMemoryUsage;

		std::map<int,LMemory> _memoryMap;

		std::vector<long> _changes;
		std::vector<timeval> _timeStamps;
		long _max;
};


#endif /* GPUMEMORYMANAGER_H_ */
