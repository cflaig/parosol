/*
 * OtherKernels.h
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

#ifndef OTHERKERNELS_H_
#define OTHERKERNELS_H_

#include "DispVector.h"
#include "DispVectorS.h"
#include "Diagonal.h"
#include "DiagonalS.h"
#include "DiagonalNComp.h"
#include "BoundaryCondition.h"

void MemsetK(t_index numStarts,t_double *dev_a,t_double val,cudaStream_t stream = 0);

void ResetBoundariesK(BoundaryCondition &bc,DispVector &y,cudaStream_t stream = 0);

void SetBoundariesK(BoundaryCondition &bc,DispVector &x,DispVector &y,cudaStream_t stream = 0);

void SetDiaK(BoundaryCondition &bc,Diagonal &dia,cudaStream_t stream = 0);

void ADiaK(DispVector &x,Diagonal &dia,cudaStream_t stream = 0);

void SetBoundaries2K(BoundaryCondition &bc,DispVector &y,cudaStream_t stream = 0);

void SetBoundaries3K(BoundaryCondition &bc,DispVector &y,cudaStream_t stream = 0);

void AddBoundariesK(BoundaryCondition &bc,DispVector &y,cudaStream_t stream = 0);

void GatherSendBufferK(t_index numStarts,t_index *dev_inds,t_double *dev_x,t_double *dev_buff);

void AddInBufferK(t_index numStarts,t_index *dev_inds,t_double *dev_x,t_double *dev_buff);

void AddInBufferCompK(t_index numStarts,t_index *dev_inds,t_double *dev_x,t_double *dev_buff);

#endif /* OTHERKERNELS_H_ */
