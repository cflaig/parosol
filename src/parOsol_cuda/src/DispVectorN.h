/*
 * DispVectorN.h
 *
 *  Created on: 27.06.2013
 *      Author: daniel
 */

#ifndef DISPVECTORN_H_
#define DISPVECTORN_H_

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
class DispVectorN : public DispVector {

public:

	DispVectorN(SyncData &syncData,GPUMemoryManager *gpuMM);

	DispVectorN(SyncData &syncData,GPUMemoryManager *gpuMM,t_double *gVec);

	~DispVectorN();

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
	 * @brief Copies the buffer from the GPU to the CPU.
	 */
	void CopyBufferToCpu();

	/**
	 * @brief Copies the buffer from the CPU to the GPU.
	 */
	void CopyBufferToGpu();

	/**
	 * @brief Copies the tail from the GPU to the CPU.
	 */
	void CopyTailToCpu();

	/**
	 * @brief Copies the tail from the CPU to the GPU.
	 */
	void CopyTailToGpu();

	t_double *gpuVec;
	t_index _gpuVecSize;
	bool _sharedVec;

};

#endif /* DISPVECTORN_H_ */
