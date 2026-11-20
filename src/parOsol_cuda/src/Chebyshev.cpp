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

#include <iostream>

#include <cuda_runtime.h>
#include "Chebyshev.h"
#include "est_ev.h"
#include "Jacobi.h"

Chebyshev::Chebyshev(StiffnessMatrix &M, int degree, bool ZeroStart,DispVector &m1,DispVector &m2,GPUMemoryManager *gpuMM, int maxit, double lmax, double lmin, double ratio, bool adaptive):
	_mat(M),
	_ldofs(M.GetNrDofs()),
	_degree(degree),
	_lmax(lmax),
	_lmin(lmin),
	_zeroStart(ZeroStart),
	z(m1),
	p(m2)
	{
	M.GenerateDiagonal();
	Solver::_gpuMemoryManager = gpuMM;
	idia = _mat.dia;
	//idia2 = _mat.dia2;
	if (maxit != 0) {
		eig e;
		if (adaptive) e = EstimateChebyshevSpectrum(_mat, maxit, z, p, gpuMM);
		else {
			Jacobi prec(_mat,gpuMM);
			e = est_ev(_mat, prec, maxit, z, p, gpuMM);
		}
		_lmax = e.large*1.05;
		_lmin = e.small;
		//use ration if set
		if (ratio >0)
			_lmin = e.large/ratio;
		if (0 == _mat.GetPID())
			std::cout << "lmax = " << _lmax << "   lmin = " << e.small << "   lmax used = " << _lmin  << std::endl;
	}
}

Chebyshev::Chebyshev(Chebyshev &old, bool ZeroStart):
  _mat(old._mat),
  idia(old.idia),
  _ldofs(old._ldofs),
  _degree(old._degree),
  _lmax(old._lmax),
  _lmin(old._lmin),
  _zeroStart(ZeroStart),
  z(old.z),
  p(old.p)
{
	Solver::_gpuMemoryManager = old._gpuMemoryManager;
}

int Chebyshev::Solve(DispVector &b,DispVector &x,cudaStream_t streamB,cudaStream_t streamX) {
  
	CCS(cudaStreamSynchronize(streamB),"Solve 1");
	CCS(cudaStreamSynchronize(streamX),"Solve 2");

	double alpha = _lmin;
	double beta = _lmax;
	double delta=2.0/ (beta - alpha);
	double theta=(beta + alpha)/2;
	double s1 = theta *delta;
	double invTheta = 1.0 / theta;

	if (_zeroStart) {
		CWiseProductSVector(invTheta,*idia,b,p);
		x.Copy(p);		//x = p;

	} else {
		_mat.Apply(x,z,streamB,streamX);
		CWiseProductS2Vector(invTheta,*idia,b,z,p,x,streamX);
	}

	double rhok = 1.0/s1, rhokp1;
	double dtmp1, dtmp2;
	if (b._syncData.switching) {
		for (int i = 1; i < _degree; i++) {
			rhokp1 = 1.0/ (2.0*s1 -rhok);
			dtmp1 = rhokp1 * rhok;
			dtmp2 = 2.0 * rhokp1 * delta;
			rhok = rhokp1;
			if (i == 1) {
				_mat.ApplyCheb(true,false,x,z,dtmp2,dtmp1,*idia,b,p,streamB,streamX);
			} else if (i + 1 == _degree) {
				_mat.ApplyCheb(false,true,x,z,dtmp2,dtmp1,*idia,b,p,streamB,streamX);
			} else {
				_mat.ApplyCheb(false,false,x,z,dtmp2,dtmp1,*idia,b,p,streamB,streamX);
			}
		}
	} else {
		for (int i = 1; i < _degree; i++) {
			_mat.Apply(x,z,streamB,streamX);
			rhokp1 = 1.0/ (2.0*s1 -rhok);
			dtmp1 = rhokp1 * rhok;
			dtmp2 = 2.0 * rhokp1 * delta;
			rhok = rhokp1;
			CWiseProductS4Vector(dtmp2,dtmp1,*idia,b,z,p,x,streamX);
		}
	}

	CCS(cudaStreamSynchronize(streamB),"Solve 3");
	CCS(cudaStreamSynchronize(streamX),"Solve 4");
	return 1;
}
