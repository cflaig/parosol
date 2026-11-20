/*
 * VectorOperations.h
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

#ifndef VECTOROPERATIONS_H_
#define VECTOROPERATIONS_H_

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include "Config.h"
#include "VectorKernels.h"
#ifdef MMX
#include <immintrin.h>
#endif

//a = a + b
void AddVector(DispVector &a,DispVector &b,cudaStream_t stream = 0);

//a = a - b
void SubtractVector(DispVector &a,DispVector &b,cudaStream_t stream = 0);

//b = a - b
void SubtractVector2(DispVector &a,DispVector &b,cudaStream_t stream = 0);

//a = alpha*a
void ScaleVector(t_double alpha,DispVector &a,cudaStream_t stream = 0);

//y = a*x + y
void Axpy(t_double a,DispVector &x,DispVector &y,cudaStream_t stream = 0);

//x = a*x + y
void Axpy2(t_double a,DispVector &x,DispVector &y,cudaStream_t stream = 0);

//c = a*b (componentwise)
void CWiseProductVector(Diagonal &a,DispVector &b,DispVector &c,cudaStream_t stream = 0);

//c = a*b*alpha (componentwise)
void CWiseProductSVector(t_double alpha,Diagonal &a,DispVector &b,DispVector &c,cudaStream_t stream = 0);

//d = a*(b-c)*alpha, e = e + d (componentwise)
void CWiseProductS2Vector(t_double alpha,Diagonal &a,DispVector &b,DispVector &c,DispVector &d,DispVector &e,cudaStream_t stream = 0);

//d = a*(b - c)*alpha + d (componentwise)
void CWiseProductS3Vector(t_double alpha,Diagonal &a,DispVector &b,DispVector &c,DispVector &d,cudaStream_t stream = 0);

//d = a*(b - c)*alpha + d*beta, e = e + d (componentwise)
void CWiseProductS4Vector(t_double alpha,t_double beta,Diagonal &a,DispVector &b,DispVector &c,DispVector &d,DispVector &e,cudaStream_t stream = 0);

t_double Dot(DispVector &a,DispVector &b,t_double *c,t_double *dev_c,t_index blockSize,cudaStream_t stream = 0);

t_double Dot2(DispVector &a,t_double *c,t_double *dev_c,t_index blockSize,cudaStream_t stream = 0);

void DDot(DispVector &a,DispVector &b,t_double *c,t_double *dev_c,t_index blockSize,t_double &res1,t_double &res2,cudaStream_t stream = 0);

bool PInverseDia(Diagonal &dia,cudaStream_t stream = 0);

#endif /* VECTOROPERATIONS_H_ */
