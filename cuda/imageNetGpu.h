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
 


#ifndef __CUDA_IMAGE_NET_H
#define __CUDA_IMAGE_NET_H

// #ifdef __cplusplus
// extern "C" {
// #endif  // __cplusplus


#include "cudaUtility.h"
#include <stdint.h>

// gpuPreImageNet
__global__ void gpuPreImageNet( float2 scale, float4* input,
							int iWidth, float* output, int oWidth, int oHeight );


// cudaPreImageNet
cudaError_t cudaPreImageNet( float4* input, size_t inputWidth, size_t inputHeight,
				         float* output, size_t outputWidth, size_t outputHeight );

// gpuPreImageNetMean
__global__ void gpuPreImageNetMean( float2 scale, float4* input,
							int iWidth, float* output, int oWidth, int oHeight, float3 mean_value );


// cudaPreImageNetMean
cudaError_t cudaPreImageNetMean( float4* input, size_t inputWidth, size_t inputHeight,
				             float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value );


// #ifdef __cplusplus
// }
// #endif  // __cplusplus

// __CUDA_IMAGE_NET_H
#endif
