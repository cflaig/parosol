/*
 * SyncData.h
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

#ifndef SYNCDATA_H_
#define SYNCDATA_H_

class SyncData {

public:

	int level;
	bool switching;
	bool compressDia;
	t_index gpuVecSize;
	t_index dotStartSize;

	t_index bufferSize;
	t_index tailSize;

	t_double *cpuBuffer;
	t_index *cpuInds;
	t_index *gpuInds;
	t_double *gpuBuffer;

	t_double *gpuVecA;
	t_double *gpuVecB;

	t_double *gVec;

};


#endif /* SYNCDATA_H_ */
