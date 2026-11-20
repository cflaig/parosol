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


#include "JacobiSmoother.h"

int JacobiSmoother::Solve(DispVector &b,DispVector &x,cudaStream_t streamB,cudaStream_t streamX) {
	int it = 0;
	if (_zeroStart) {
		//CWiseProductSVectorK(_w,*idia,b,x);
		it = 1;
	}
	for(; it<_step;it++) {
		_mat.Apply(x,r);
		//CWiseProductS3VectorK(_w,*idia,b,r,x);
	}
	return 1;
}
