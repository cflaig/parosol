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



#ifndef STIFFNESSMATRIX_H
#define STIFFNESSMATRIX_H

#include <string>
#include "Config.h"
#include "DispVector.h"
#include "Diagonal.h"


//! An Interface which describes a MatrixVector Product.


class StiffnessMatrix 
{
public:

  //! ElementByElementMatrix destructor
  virtual ~StiffnessMatrix() {};

   /** Compute the matrix vetctor product.
   * 
   * @param x input vector
   * @param y result vector
   * @return error value
   */
  virtual int Apply (DispVector &x,DispVector &y,cudaStream_t streamX = 0,cudaStream_t streamY = 0) = 0;
  virtual int ApplyCheb (bool first,bool last,DispVector &x,DispVector &y,t_double alpha,t_double beta,Diagonal &a,DispVector &b,DispVector &d,cudaStream_t streamX = 0,cudaStream_t streamY = 0) = 0;


  /** Name of the Operator
   * 
   * @return Name in an string
   */
  virtual const std::string  Label () const = 0;

   /** Gets the Diagonal of the operator.
   * 
   * @return diagonal
   */
  virtual void GenerateDiagonal() = 0;

   /** Gets the number of degrees of freedom.
   * 
   * @return number of dofs
   */
  virtual t_index GetNrDofs() = 0;

   /** Computes the dot product of two vectors according the communicator is used in the matrix operator.
   * 
   * @param a first vector
   * @param b second vector
   * @return dot product
   */
  virtual t_double dot(DispVector &a,DispVector &b,cudaStream_t stream = 0) = 0;

  virtual t_double dot(DispVector &a,cudaStream_t stream = 0) = 0;
  
  virtual void ddot(DispVector &a,DispVector &b,t_double &dot1,t_double &dot2,cudaStream_t stream = 0) = 0;

  /** Gets the process id
   * @return PID
   */
  virtual int GetPID()=0;

  virtual void SetVectorRandom(DispVector &x) = 0;
  
  virtual BoundaryCondition & GetBC() = 0;

  virtual SyncData& GetSyncData() = 0;

  /** initializes an vector with random values according the distribution of the mesh on the computing nodes.
   * @param x vector that is filled with random numbers.
   */

  Diagonal *dia;
};

#endif /* STIFFNESSMATRIX_H */


