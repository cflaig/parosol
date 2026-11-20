/*
 * GPUGridBuilder.h
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

#ifndef GPUGRIDBUILDER_H_
#define GPUGRIDBUILDER_H_

#include <vector>
#include <mpi.h>
#include <omp.h>
#include "Config.h"
#include "ApplyKernels.h"
#include "OctreeGrid.h"
#include "GPUGridContainer.h"

template <class T>
class GPUGridBuilder {

public:

	GPUGridBuilder(GPUMemoryManager *gpuMM) {
		_gpuMM = gpuMM;
	}

	virtual ~GPUGridBuilder() {

	}

	virtual GPUGridContainer BuildGrid(OctreeGrid<T> *oGrid,int level,MlOctreeGrid<T> *mlGrid = 0) = 0;

	GPUMemoryManager *_gpuMM;
};

#endif /* GPUGRIDBUILDER_H_ */
