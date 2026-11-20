/*
 * PCGSolverMC.cpp
 *
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


#include "PCGSolverMC.h"

PCGSolverMC::PCGSolverMC(StiffnessMatrix &M, Solver &precon,DispVector &dx,DispVector &dd, double tol, int maxIter, bool verbose, int freq):
	_mat(M),
	_prec(precon),
	_maxIter(maxIter),
	_tol(tol),
	_verbose(verbose),
	_freq(freq),
	x(dx),
	d(dd)
	{
	CCS(cudaHostAlloc((void**) &t_x,x._syncData.gpuVecSize*3*sizeof(t_double),cudaHostAllocDefault),"PCGSlverMC 1");
	CCS(cudaHostAlloc((void**) &t_d,d._syncData.gpuVecSize*3*sizeof(t_double),cudaHostAllocDefault),"PCGSlverMC 2");
	_first = true;
}

PCGSolverMC::~PCGSolverMC() {
	CCS(cudaFreeHost(t_x),"~PCGSolverMC 1");
	CCS(cudaFreeHost(t_d),"~PCGSolverMC 2");
}

int PCGSolverMC::Solve(DispVector &b,DispVector &tx,cudaStream_t streamB,cudaStream_t streamX) {

	int MyPID = _mat.GetPID();
	std::ios_base::fmtflags origflag;
	origflag = std::cout.flags();
	if (_verbose) {
	std::cout.setf(std::ios::scientific);
	}
	double alpha, res0, delta_new, delta_old, beta,norm;
	int i = 0;

	_mat.Apply(tx,d,streamB,streamX);

	tx.CopyToExtCpuVectorAsync(t_x,0);

	//residual
	SubtractVector(b,d);

	tx.Copy(d);

	_prec.Solve(b,tx,streamB,streamX);

	d.Copy(tx);
	x.CopyFromExtCpuVectorAsync(t_x,0);

	DispVector &s = tx;
	s.Copy(b);

	_mat.ddot(b,d,res0,norm);
	delta_new = res0;
	res0 = sqrt(res0);
	norm = sqrt(norm);
	double resreal0 = _residualReference > 0.0 ? _residualReference : norm;

	if(_verbose) {
		PCOUT(MyPID, "Initial Residuum: "<< resreal0 << std::endl)
	}
	while ( i < _maxIter && norm > 0.0 && norm/resreal0 > _tol) {
		if (_verbose && i % _freq == 0) {
			PCOUT(MyPID, "Iteration: "<< i << "\t Residuum precondioned: " << sqrt(delta_new)/res0)
			PCOUT(MyPID, "\t Residuum: " << norm/resreal0 << std::endl)
		}

		_mat.Apply(d,s,streamB,streamX);
		d.CopyToExtCpuVectorAsync(t_d,streamB);
		alpha = delta_new/_mat.dot(d,s,streamX);

		Axpy(alpha,d,x,streamX);
		x.CopyToExtCpuVectorAsync(t_x,streamX);
		Axpy((-1)*alpha,s,b,streamB);

		_prec.Solve(b,s,streamB,streamX);

		d.CopyFromExtCpuVectorAsync(t_d,streamB);

		delta_old = delta_new;
		_mat.ddot(b,s,delta_new,norm,streamX);
		norm = sqrt(norm);

		beta = delta_new/delta_old;
		Axpy2(beta,d,s,streamB);
		i++;
		x.CopyFromExtCpuVectorAsync(t_x,streamX);
	}
	if (_verbose) {
		PCOUT(MyPID, "Total iterations: "<< i << "\t Residuum preconditioned: " << sqrt(delta_new)/res0 << std::endl)
		PCOUT(MyPID, "Total iterations: "<< i << "\t Residuum:                " << norm/resreal0 << std::endl);
		std::cout.flags(origflag);
    }

	tx.Copy(x);

	return i;
}

const std::string PCGSolverMC::Label() const
{
  return("Preconditioned CG-Solver\n");
}
