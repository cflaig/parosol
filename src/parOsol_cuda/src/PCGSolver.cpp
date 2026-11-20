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


#include "PCGSolver.h"
#include "DispVectorN.h"

PCGSolver::PCGSolver(StiffnessMatrix &M, Solver &precon, double tol, int maxIter, bool verbose, int freq):
  _mat(M),
  _prec(precon),
  _maxIter(maxIter),
  _tol(tol),
  _verbose(verbose),
  _freq(freq) {
	_dispVecsInitialized = false;
	_first = true;
}

PCGSolver::~PCGSolver() {
	if (_dispVecsInitialized) {
		delete d;
		delete s;
	}
}

int PCGSolver::Solve(DispVector &b,DispVector &x,cudaStream_t streamB,cudaStream_t streamX) {

	if (!_dispVecsInitialized) {
		if (_mat.GetSyncData().switching) {
			d = new DispVectorS(_mat.GetSyncData(),_gpuMemoryManager);
			s = new DispVectorS(_mat.GetSyncData(),_gpuMemoryManager);
		} else {
			d = new DispVectorN(_mat.GetSyncData(),_gpuMemoryManager);
			s = new DispVectorN(_mat.GetSyncData(),_gpuMemoryManager);
		}
		_dispVecsInitialized = true;
	}

	int MyPID = _mat.GetPID();
	std::ios_base::fmtflags origflag;
	origflag = std::cout.flags();
	if (_verbose) {
		std::cout.setf(std::ios::scientific);
	}
	double alpha, res0, delta_new, delta_old, beta,norm;
	int i = 0;

	_mat.Apply(x,*d,streamB,streamX);

	//residual

	SubtractVector(b,*d);

	_prec.Solve(b,*d,streamB,streamX);

	s->Copy(b);

	res0 = delta_new = _mat.dot(b,*d);
	res0 = sqrt(res0);
	double resreal0 = sqrt(_mat.dot(b));
	norm = resreal0;
	if(_verbose) {
		PCOUT(MyPID, "Initial Residuum: "<< resreal0 << std::endl)
	}
	while ( i < _maxIter && norm/resreal0 > _tol) {
		if (_verbose && i % _freq == 0) {
			PCOUT(MyPID, "Iteration: "<< i << "\t Residuum precondioned: " << sqrt(delta_new)/res0)
			PCOUT(MyPID, "\t Residuum: " << norm/resreal0 << std::endl)
		}

		_mat.Apply(*d,*s,streamB,streamX);

		alpha = delta_new/_mat.dot(*d,*s);

		Axpy(alpha,*d,x);

		Axpy((-1)*alpha,*s,b);

		_prec.Solve(b,*s,streamB,streamX);

		delta_old = delta_new;
		delta_new = _mat.dot(b,*s);

		beta = delta_new/delta_old;
		Axpy2(beta,*d,*s);

		i++;
		norm =  sqrt(_mat.dot(b));

	}
	if (_verbose) {
		PCOUT(MyPID, "Total iterations: "<< i << "\t Residuum preconditioned: " << sqrt(delta_new)/res0 << std::endl)
		PCOUT(MyPID, "Total iterations: "<< i << "\t Residuum:                " << norm/resreal0 << std::endl);
		std::cout.flags(origflag);
    }

	return i;
}

const std::string PCGSolver::Label() const {
	return("Preconditioned CG-Solver\n");
}
