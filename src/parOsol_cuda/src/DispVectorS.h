/*
 * DispVectorS.h
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

#ifndef DISPVECTORS_H_
#define DISPVECTORS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "Config.h"
#include "DispVector.h"
#include "SyncData.h"
#include "GPUMemoryManager.h"

/*!
 * The DispVector is the representation of the displacement vector, combining the parts on the CPU and the GPU.
 * Additionally it provides methods to transfer data between the CPU and GPU-parts of the vector
 */
class DispVectorS : public DispVector {

public:

	DispVectorS(SyncData &syncData,GPUMemoryManager *gpuMM);

	~DispVectorS();

	t_double* GetVector();

	/**
	 * @brief Sets all values in the vector to val.
	 * @param[in] val The value the vector is set to.
	 */
	void Memset(t_double val,cudaStream_t stream = 0);

	/**
	 * @brief Copies the contents of the given DispVector in this DispVector.
	 * @param[in] vec The DispVector that is copied.
	 */
	void Copy(DispVector &vec);

	/**
	 * @brief Copies the contents of this DispVector to an array on the CPU.
	 * @param[in] vec A pointer to an array on the CPU.
	 */
	void CopyToExtCpuVector(t_double *vec);

	/**
	 * @brief Inserts data from an array on the CPU.
	 * @param[in] vec A pointer to an array on the CPU.
	 */
	void CopyFromExtCpuVector(t_double *vec);

	/**
	* @brief Copies the contents of this DispVector to an array on the CPU asynchronously.
	* @param[in] vec A pointer to an array on the CPU.
	* @param[in] stream The CudaStream to be used.
	*/
	void CopyToExtCpuVectorAsync(t_double *vec,cudaStream_t stream);

	/**
	 * @brief Inserts data from an array on the CPU asynchronously.
	 * @param[in] vec A pointer to an array on the CPU.
	 * @param[in] stream The CudaStream to be used.
	 */
	void CopyFromExtCpuVectorAsync(t_double *vec,cudaStream_t stream);

	void PrepareIn(cudaStream_t stream);
	void PrepareOut(cudaStream_t stream,bool vecIn = false);
	void PostProcess(cudaStream_t stream);

	/**
	 * @brief Copies the tail from the GPU to the CPU.
	 */
	void CopyTailToCpu();

	/**
	 * @brief Copies the tail from the CPU to the GPU.
	 */
	void CopyTailToGpu();

	t_double *cpuVec;
	t_index _cpuVecSize;

	bool _vecIn;

};

#endif /* DISPVECTORS_H_ */
