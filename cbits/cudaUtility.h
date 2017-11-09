/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __CUDA_UTILITY_H_
#define __CUDA_UTILITY_H_

// #ifdef __cplusplus
// extern "C" {
// #endif  // __cplusplus

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <string.h>


/**
 * Execute a CUDA call and print out any errors
 * @return the original cudaError_t result
 * @ingroup util
 */
#define CUDA(x)				cudaCheckError((x), #x, __FILE__, __LINE__)

/**
 * Evaluates to true on success
 * @ingroup util
 */
#define CUDA_SUCCESS(x)			(CUDA(x) == cudaSuccess)

/**
 * Evaluates to true on failure
 * @ingroup util
 */
#define CUDA_FAILED(x)			(CUDA(x) != cudaSuccess)

/**
 * Return from the boolean function if CUDA call fails
 * @ingroup util
 */
#define CUDA_VERIFY(x)			if(CUDA_FAILED(x))	return false;

/**
 * LOG_CUDA string.
 * @ingroup util
 */
#define LOG_CUDA "[cuda]   "

/*
 * define this if you want all cuda calls to be printed
 */
//#define CUDA_TRACE



/**
 * cudaCheckError
 * @ingroup util
 */
inline cudaError_t cudaCheckError(cudaError_t retval, const char* txt, const char* file, int line )
{
#if !defined(CUDA_TRACE)
	if( retval == cudaSuccess)
		return cudaSuccess;
#endif

	//int activeDevice = -1;
	//cudaGetDevice(&activeDevice);

	//Log("[cuda]   device %i  -  %s\n", activeDevice, txt);
	
	printf(LOG_CUDA "%s\n", txt);


	if( retval != cudaSuccess )
	{
		printf(LOG_CUDA "   %s (error %u) (hex 0x%02X)\n", cudaGetErrorString(retval), retval, retval);
		printf(LOG_CUDA "   %s:%i\n", file, line);	
	}

	return retval;
}

/**
 * Allocate ZeroCopy mapped memory, shared between CUDA and CPU.
 * @ingroup util
 */
inline bool cudaAllocMapped( void** cpuPtr, void** gpuPtr, size_t size )
{
	if( !cpuPtr || !gpuPtr || size == 0 )
		return false;

	//CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));

	if( CUDA_FAILED(cudaHostAlloc(cpuPtr, size, cudaHostAllocMapped)) )
		return false;

	if( CUDA_FAILED(cudaHostGetDevicePointer(gpuPtr, *cpuPtr, 0)) )
		return false;

	memset(*cpuPtr, 0, size);
	printf("[cuda]  cudaAllocMapped %zu bytes, CPU %p GPU %p\n", size, *cpuPtr, *gpuPtr);
	return true;
}

// #ifdef __cplusplus
// }
// #endif  // __cplusplus

#endif
