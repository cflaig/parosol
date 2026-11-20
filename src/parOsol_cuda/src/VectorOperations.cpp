/*
 * VectorOperations.cpp
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

#include "VectorOperations.h"

//a = a + b
void AddVector(DispVector &a,DispVector &b,cudaStream_t stream) {
	if (a._syncData.switching) {
		DispVectorS *aS = (DispVectorS*) &a;
		DispVectorS *bS = (DispVectorS*) &b;
#pragma omp parallel for
#ifdef MMX
		for (uint i = 0;i < aS->_cpuVecSize;i = i + 4) {
			__m256d amm = _mm256_load_pd(&aS->cpuVec[i]);
			__m256d bmm = _mm256_load_pd(&bS->cpuVec[i]);
			amm = _mm256_add_pd(amm,bmm);
			_mm256_store_pd(&aS->cpuVec[i],amm);
		}
#else
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			aS->cpuVec[i] += bS->cpuVec[i];
		}
#endif
	} else {
		tAddVectorK(a,b,stream);
	}
}

//a = a - b
void SubtractVector(DispVector &a,DispVector &b,cudaStream_t stream) {
	if (a._syncData.switching) {
		DispVectorS *aS = (DispVectorS*) &a;
		DispVectorS *bS = (DispVectorS*) &b;
#pragma omp parallel for
#ifdef MMX
		for (uint i = 0;i < aS->_cpuVecSize;i = i + 4) {
			__m256d amm = _mm256_load_pd(&aS->cpuVec[i]);
			__m256d bmm = _mm256_load_pd(&bS->cpuVec[i]);
			amm = _mm256_sub_pd(amm,bmm);
			_mm256_store_pd(&aS->cpuVec[i],amm);
		}
#else
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			aS->cpuVec[i] -= bS->cpuVec[i];
		}
#endif
	} else {
		tSubtractVectorK(a,b,stream);
	}
}

//b = a - b
void SubtractVector2(DispVector &a,DispVector &b,cudaStream_t stream) {
	if (a._syncData.switching) {
		DispVectorS *aS = (DispVectorS*) &a;
		DispVectorS *bS = (DispVectorS*) &b;
#pragma omp parallel for
#ifdef MMX
		for (uint i = 0;i < aS->_cpuVecSize;i = i + 4) {
			__m256d amm = _mm256_load_pd(&aS->cpuVec[i]);
			__m256d bmm = _mm256_load_pd(&bS->cpuVec[i]);
			amm = _mm256_sub_pd(amm,bmm);
			_mm256_store_pd(&bS->cpuVec[i],amm);
		}
#else
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			bS->cpuVec[i] = aS->cpuVec[i] - bS->cpuVec[i];
		}
#endif
	} else {
		tSubtractVector2K(a,b,stream);
	}
}

//a = alpha*a
void ScaleVector(t_double alpha,DispVector &a,cudaStream_t stream) {
	if (a._syncData.switching) {
		DispVectorS *aS = (DispVectorS*) &a;
#pragma omp parallel for
#ifdef MMX
		for (uint i = 0;i < aS->_cpuVecSize;i = i + 4) {
			__m256d amm = _mm256_load_pd(&aS->cpuVec[i]);
			__m256d alphamm = _mm256_broadcast_sd(&alpha);
			amm = _mm256_mul_pd(amm,alphamm);
			_mm256_store_pd(&aS->cpuVec[i],amm);
		}
#else
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			aS->cpuVec[i] *= alpha;
		}
#endif
	} else {
		tScaleVectorK(alpha,a,stream);
	}
}

//y = a*x + y
void Axpy(t_double a,DispVector &x,DispVector &y,cudaStream_t stream) {
	if (x._syncData.switching) {
		DispVectorS *xS = (DispVectorS*) &x;
		DispVectorS *yS = (DispVectorS*) &y;
#pragma omp parallel for
#ifdef MMX
		for (uint i = 0;i < xS->_cpuVecSize;i = i + 4) {
			__m256d xmm = _mm256_load_pd(&xS->cpuVec[i]);
			__m256d ymm = _mm256_load_pd(&yS->cpuVec[i]);
			__m256d amm = _mm256_broadcast_sd(&a);
			xmm = _mm256_mul_pd(xmm,amm);
			ymm = _mm256_add_pd(ymm,xmm);
			_mm256_store_pd(&yS->cpuVec[i],ymm);
		}
#else
		for (uint i = 0;i < xS->_cpuVecSize;i++) {
			yS->cpuVec[i] += a*xS->cpuVec[i];
		}
#endif
	} else {
		tAxpyK(a,x,y,stream);
	}
}

//x = a*x + y
void Axpy2(t_double a,DispVector &x,DispVector &y,cudaStream_t stream) {
	if (x._syncData.switching) {
		DispVectorS *xS = (DispVectorS*) &x;
		DispVectorS *yS = (DispVectorS*) &y;
#pragma omp parallel for
#ifdef MMX
		for (uint i = 0;i < xS->_cpuVecSize;i = i + 4) {
			__m256d xmm = _mm256_load_pd(&xS->cpuVec[i]);
			__m256d ymm = _mm256_load_pd(&yS->cpuVec[i]);
			__m256d amm = _mm256_broadcast_sd(&a);
			xmm = _mm256_mul_pd(xmm,amm);
			ymm = _mm256_add_pd(ymm,xmm);
			_mm256_store_pd(&xS->cpuVec[i],ymm);
		}
#else
		for (uint i = 0;i < xS->_cpuVecSize;i++) {
			xS->cpuVec[i] = a*xS->cpuVec[i] + yS->cpuVec[i];
		}
#endif
	} else {
		tAxpy2K(a,x,y,stream);
	}
}

//c = a*b (componentwise)
void CWiseProductVector(Diagonal &a,DispVector &b,DispVector &c,cudaStream_t stream) {
	if (a._syncData.switching) {
		DiagonalS *aS = (DiagonalS*) &a;
		DispVectorS *bS = (DispVectorS*) &b;
		DispVectorS *cS = (DispVectorS*) &c;
#pragma omp parallel for
#ifdef MMX
		for (uint i = 0;i < aS->_cpuVecSize;i = i + 4) {
			__m256d amm = _mm256_load_pd(&aS->cpuVec[i]);
			__m256d bmm = _mm256_load_pd(&bS->cpuVec[i]);
			amm = _mm256_mul_pd(amm,bmm);
			_mm256_store_pd(&cS->cpuVec[i],amm);
		}
#else
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			cS->cpuVec[i] = aS->cpuVec[i]*bS->cpuVec[i];
		}
#endif
	} else {
		tCWiseProductVectorK(a,b,c,stream);
	}
}

//c = a*b*alpha (componentwise)
void CWiseProductSVector(t_double alpha,Diagonal &a,DispVector &b,DispVector &c,cudaStream_t stream) {
	if (a._syncData.switching) {
		DiagonalS *aS = (DiagonalS*) &a;
		DispVectorS *bS = (DispVectorS*) &b;
		DispVectorS *cS = (DispVectorS*) &c;
#pragma omp parallel for
#ifdef MMX
		for (uint i = 0;i < aS->_cpuVecSize;i = i + 4) {
			__m256d amm = _mm256_load_pd(&aS->cpuVec[i]);
			__m256d bmm = _mm256_load_pd(&bS->cpuVec[i]);
			__m256d alphamm = _mm256_broadcast_sd(&alpha);
			amm = _mm256_mul_pd(amm,bmm);
			amm = _mm256_mul_pd(amm,alphamm);
			_mm256_store_pd(&cS->cpuVec[i],amm);
		}
#else
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			cS->cpuVec[i] = aS->cpuVec[i]*bS->cpuVec[i]*alpha;
		}
#endif
	} else {
		tCWiseProductSVectorK(alpha,a,b,c,stream);
	}
}

//d = a*(b-c)*alpha, e = e + d (componentwise)
void CWiseProductS2Vector(t_double alpha,Diagonal &a,DispVector &b,DispVector &c,DispVector &d,DispVector &e,cudaStream_t stream) {
	if (a._syncData.switching) {
		DiagonalS *aS = (DiagonalS*) &a;
		DispVectorS *bS = (DispVectorS*) &b;
		DispVectorS *cS = (DispVectorS*) &c;
		DispVectorS *dS = (DispVectorS*) &d;
		DispVectorS *eS = (DispVectorS*) &e;
#pragma omp parallel for
#ifdef MMX
		for (uint i = 0;i < aS->_cpuVecSize;i = i + 4) {
			__m256d amm = _mm256_load_pd(&aS->cpuVec[i]);
			__m256d bmm = _mm256_load_pd(&bS->cpuVec[i]);
			__m256d cmm = _mm256_load_pd(&cS->cpuVec[i]);
			__m256d alphamm = _mm256_broadcast_sd(&alpha);
			amm = _mm256_mul_pd(amm,alphamm);
			bmm = _mm256_sub_pd(bmm,cmm);
			amm = _mm256_mul_pd(amm,bmm);
			_mm256_store_pd(&dS->cpuVec[i],amm);
			__m256d emm = _mm256_load_pd(&eS->cpuVec[i]);
			emm = _mm256_add_pd(emm,amm);
			_mm256_store_pd(&eS->cpuVec[i],emm);
		}
#else
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			dS->cpuVec[i] = aS->cpuVec[i]*(bS->cpuVec[i] - cS->cpuVec[i])*alpha;
			eS->cpuVec[i] += dS->cpuVec[i];
		}
#endif
	} else {
		tCWiseProductS2VectorK(alpha,a,b,c,d,e,stream);
	}
}

//d = a*(b - c)*alpha + d (componentwise)
void CWiseProductS3Vector(t_double alpha,Diagonal &a,DispVector &b,DispVector &c,DispVector &d,cudaStream_t stream) {
	if (a._syncData.switching) {
		DiagonalS *aS = (DiagonalS*) &a;
		DispVectorS *bS = (DispVectorS*) &b;
		DispVectorS *cS = (DispVectorS*) &c;
		DispVectorS *dS = (DispVectorS*) &d;
#pragma omp parallel for
#ifdef MMX
		for (uint i = 0;i < aS->_cpuVecSize;i = i + 4) {
			__m256d amm = _mm256_load_pd(&aS->cpuVec[i]);
			__m256d bmm = _mm256_load_pd(&bS->cpuVec[i]);
			__m256d cmm = _mm256_load_pd(&cS->cpuVec[i]);
			__m256d dmm = _mm256_load_pd(&dS->cpuVec[i]);
			__m256d alphamm = _mm256_broadcast_sd(&alpha);
			amm = _mm256_mul_pd(amm,alphamm);
			bmm = _mm256_sub_pd(bmm,cmm);
			amm = _mm256_mul_pd(amm,bmm);
			dmm = _mm256_add_pd(dmm,amm);
			_mm256_store_pd(&dS->cpuVec[i],dmm);
		}
#else
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			dS->cpuVec[i] += aS->cpuVec[i]*(bS->cpuVec[i] - cS->cpuVec[i])*alpha;
		}
#endif
	} else {
		tCWiseProductS3VectorK(alpha,a,b,c,d,stream);
	}
}

//d = a*(b - c)*alpha + d*beta, e = e + d (componentwise)
void CWiseProductS4Vector(t_double alpha,t_double beta,Diagonal &a,DispVector &b,DispVector &c,DispVector &d,DispVector &e,cudaStream_t stream) {
	if (a._syncData.switching) {
		DiagonalS *aS = (DiagonalS*) &a;
		DispVectorS *bS = (DispVectorS*) &b;
		DispVectorS *cS = (DispVectorS*) &c;
		DispVectorS *dS = (DispVectorS*) &d;
		DispVectorS *eS = (DispVectorS*) &e;
#pragma omp parallel for
#ifdef MMX
		for (uint i = 0;i < aS->_cpuVecSize;i = i + 4) {
			__m256d amm = _mm256_load_pd(&aS->cpuVec[i]);
			__m256d bmm = _mm256_load_pd(&bS->cpuVec[i]);
			__m256d cmm = _mm256_load_pd(&cS->cpuVec[i]);
			__m256d dmm = _mm256_load_pd(&dS->cpuVec[i]);
			__m256d alphamm = _mm256_broadcast_sd(&alpha);
			__m256d betamm = _mm256_broadcast_sd(&beta);
			amm = _mm256_mul_pd(amm,alphamm);
			bmm = _mm256_sub_pd(bmm,cmm);
			amm = _mm256_mul_pd(amm,bmm);
			dmm = _mm256_mul_pd(dmm,betamm);
			dmm = _mm256_add_pd(dmm,amm);
			_mm256_store_pd(&dS->cpuVec[i],dmm);
			__m256d emm = _mm256_load_pd(&eS->cpuVec[i]);
			emm = _mm256_add_pd(emm,dmm);
			_mm256_store_pd(&eS->cpuVec[i],emm);
		}
#else
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			dS->cpuVec[i] = aS->cpuVec[i]*(bS->cpuVec[i] - cS->cpuVec[i])*alpha + dS->cpuVec[i]*beta;
			eS->cpuVec[i] += dS->cpuVec[i];
		}
#endif
	} else {
		tCWiseProductS4VectorK(alpha,beta,a,b,c,d,e,stream);
	}
}

t_double Dot(DispVector &a,DispVector &b,t_double *c,t_double *dev_c,t_index blockSize,cudaStream_t stream) {
	int _numThreads;
	#pragma omp parallel
	if (omp_get_thread_num() == 0) {
		_numThreads = omp_get_num_threads();
	}

	t_double res = 0;
	if (a._syncData.switching) {
		DispVectorS *aS = (DispVectorS*) &a;
		DispVectorS *bS = (DispVectorS*) &b;
		t_double results[16];
#ifdef MMX
#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			uint start = tid*aS->_cpuVecSize/_numThreads;
			if (start % 4 != 0) {
				start += 4 - (start % 4);
			}
			uint end = (tid + 1)*aS->_cpuVecSize/_numThreads;
			if (end % 4 != 0) {
				end += 4 - (end % 4);
			} else if (tid + 1 == _numThreads) {
				end = aS->_cpuVecSize;
			}
			__m256d rmm = _mm256_set_pd(0,0,0,0);
			uint i = start;
			for (;i + 4 <= end;i = i + 4) {
				__m256d amm = _mm256_load_pd(&aS->cpuVec[i]);
				__m256d bmm = _mm256_load_pd(&bS->cpuVec[i]);
				amm = _mm256_mul_pd(amm,bmm);
				rmm = _mm256_add_pd(rmm,amm);
			}
			results[tid] = 0;
			for (;i < end;i++) {
				results[tid] += aS->cpuVec[i]*bS->cpuVec[i];
			}
			_mm256_store_pd(&c[tid*4],rmm);

			for (int i = 0;i < 4;i++) {
				results[tid] += c[tid*4 + i];
			}
		}
		for (int i = 0;i < _numThreads;i++) {
			res += results[i];
		}
#else
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			res += aS->cpuVec[i]*bS->cpuVec[i];
		}
#endif
	} else {
		res = tDotK(a,b,c,dev_c,blockSize,stream);
	}
	return res;
}

t_double Dot2(DispVector &a,t_double *c,t_double *dev_c,t_index blockSize,cudaStream_t stream) {
	int _numThreads;
	#pragma omp parallel
	if (omp_get_thread_num() == 0) {
		_numThreads = omp_get_num_threads();
	}

	t_double res = 0;
	if (a._syncData.switching) {
		DispVectorS *aS = (DispVectorS*) &a;
		t_double results[16];
#ifdef MMX
#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			uint start = tid*aS->_cpuVecSize/_numThreads;
			if (start % 4 != 0) {
				start += 4 - (start % 4);
			}
			uint end = (tid + 1)*aS->_cpuVecSize/_numThreads;
			if (end % 4 != 0) {
				end += 4 - (end % 4);
			} else if (tid + 1 == _numThreads) {
				end = aS->_cpuVecSize;
			}
			__m256d rmm = _mm256_set_pd(0,0,0,0);
			uint i = start;
			for (;i + 4 <= end;i = i + 4) {
				__m256d amm = _mm256_load_pd(&aS->cpuVec[i]);
				amm = _mm256_mul_pd(amm,amm);
				rmm = _mm256_add_pd(rmm,amm);
			}
			results[tid] = 0;
			for (;i < end;i++) {
				results[tid] += aS->cpuVec[i]*aS->cpuVec[i];
			}
			_mm256_store_pd(&c[tid*4],rmm);
			for (int i = 0;i < 4;i++) {
				results[tid] += c[tid*4 + i];
			}
		}
		for (int i = 0;i < _numThreads;i++) {
			res += results[i];
		}
#else
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			res += aS->cpuVec[i]*aS->cpuVec[i];
		}
#endif
	} else {
		res = tDot2K(a,c,dev_c,blockSize,stream);
	}
	return res;
}

void DDot(DispVector &a,DispVector &b,t_double *c,t_double *dev_c,t_index blockSize,t_double &res1,t_double &res2,cudaStream_t stream) {
	int _numThreads;
	#pragma omp parallel
	if (omp_get_thread_num() == 0) {
		_numThreads = omp_get_num_threads();
	}

	if (a._syncData.switching) {
		DispVectorS *aS = (DispVectorS*) &a;
		DispVectorS *bS = (DispVectorS*) &b;
		t_double results[32];
#ifdef MMX
#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			uint start = tid*aS->_cpuVecSize/_numThreads;
			if (start % 4 != 0) {
				start += 4 - (start % 4);
			}
			uint end = (tid + 1)*aS->_cpuVecSize/_numThreads;
			if (end % 4 != 0) {
				end += 4 - (end % 4);
			} else if (tid + 1 == _numThreads) {
				end = aS->_cpuVecSize;
			}
			__m256d r1mm = _mm256_set_pd(0,0,0,0);
			__m256d r2mm = _mm256_set_pd(0,0,0,0);
			uint i = start;
			for (;i + 4 <= end;i = i + 4) {
				__m256d amm = _mm256_load_pd(&aS->cpuVec[i]);
				__m256d bmm = _mm256_load_pd(&bS->cpuVec[i]);
				bmm = _mm256_mul_pd(amm,bmm);
				amm = _mm256_mul_pd(amm,amm);
				r1mm = _mm256_add_pd(r1mm,bmm);
				r2mm = _mm256_add_pd(r2mm,amm);
			}
			results[2*tid] = 0;
			results[2*tid + 1] = 0;
			for (;i < end;i++) {
				results[2*tid] += aS->cpuVec[i]*bS->cpuVec[i];
				results[2*tid + 1] += aS->cpuVec[i]*aS->cpuVec[i];
			}
			_mm256_store_pd(&c[tid*8],r1mm);
			_mm256_store_pd(&c[tid*8 + 4],r2mm);

			for (int i = 0;i < 4;i++) {
				results[2*tid] += c[tid*8 + i];
				results[2*tid + 1] += c[tid*8 + 4 + i];
			}
		}
		res1 = 0;
		res2 = 0;
		for (int i = 0;i < _numThreads;i++) {
			res1 += results[2*i];
			res2 += results[2*i + 1];
		}
#else
		for (uint i = 0;i < aS->_cpuVecSize;i++) {
			res1 += aS->cpuVec[i]*bS->cpuVec[i];
			res2 += aS->cpuVec[i]*aS->cpuVec[i];
		}
#endif
	} else {
		res1 = tDotK(a,b,c,dev_c,blockSize,stream);
		res2 = tDot2K(a,c,dev_c,blockSize,stream);
	}
}

bool PInverseDia(Diagonal &dia,cudaStream_t stream) {
	bool hasZeros = false;
	if (dia._syncData.switching) {
		DiagonalS *diaS = (DiagonalS*) &dia;
#pragma omp parallel for
		for (uint i = 0;i < diaS->_cpuVecSize;i++) {
			if (diaS->cpuVec[i] != 0) {
				diaS->cpuVec[i] = 1.0/diaS->cpuVec[i];
			} else {
				hasZeros = true;
			}
		}
	} else {
		tPInverseDiaK(dia,stream);
	}
	return hasZeros;
}
