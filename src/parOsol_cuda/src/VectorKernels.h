/*
 * VectorKernels.h
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

#ifndef VECTORKERNELS_H_
#define VECTORKERNELS_H_

#include "DispVector.h"
#include "DispVectorS.h"
#include "DispVectorN.h"
#include "Diagonal.h"
#include "DiagonalS.h"
#include "DiagonalN.h"
#include "DiagonalNComp.h"
#include "BoundaryCondition.h"

/**
 * @brief a = a + b
 */
void tAddVectorK(DispVector &a,DispVector &b,cudaStream_t stream = 0);

/**
 * @brief a = a - b
 */
void tSubtractVectorK(DispVector &a,DispVector &b,cudaStream_t stream = 0);

/**
 * @brief b = a - b
 */
void tSubtractVector2K(DispVector &a,DispVector &b,cudaStream_t stream = 0);

/**
 * @brief a = alpha*a
 */
void tScaleVectorK(t_double alpha,DispVector &a,cudaStream_t stream = 0);

/**
 * @brief y = a*x + y
 */
void tAxpyK(t_double a,DispVector &x,DispVector &y,cudaStream_t stream = 0);

/**
 * @brief x = a*x + y
 */
void tAxpy2K(t_double a,DispVector &x,DispVector &y,cudaStream_t stream = 0);

/**
 * @brief c = a*b (componentwise)
 */
void tCWiseProductVectorK(Diagonal &a,DispVector &b,DispVector &c,cudaStream_t stream = 0);

/**
 * @brief c = a*b*alpha (componentwise)
 */
void tCWiseProductSVectorK(t_double alpha,Diagonal &a,DispVector &b,DispVector &c,cudaStream_t stream = 0);

/**
 * @brief d = a*(b-c)*alpha, e = e + d (componentwise)
 */
void tCWiseProductS2VectorK(t_double alpha,Diagonal &a,DispVector &b,DispVector &c,DispVector &d,DispVector &e,cudaStream_t stream = 0);

/**
 * @brief d = a*(b - c)*alpha + d (componentwise)
 */
void tCWiseProductS3VectorK(t_double alpha,Diagonal &a,DispVector &b,DispVector &c,DispVector &d,cudaStream_t stream = 0);

/**
 * @brief d = a*(b - c)*alpha + d*beta, e = e + d (componentwise)
 */
void tCWiseProductS4VectorK(t_double alpha,t_double beta,Diagonal &a,DispVector &b,DispVector &c,DispVector &d,DispVector &e,cudaStream_t stream = 0);

t_double tDotK(DispVector &a,DispVector &b,t_double *c,t_double *dev_c,t_index blockSize,cudaStream_t stream = 0);

t_double tDot2K(DispVector &a,t_double *c,t_double *dev_c,t_index blockSize,cudaStream_t stream = 0);

bool tPInverseDiaK(Diagonal &dia,cudaStream_t stream = 0);

#endif /* VECTORKERNELS_H_ */
