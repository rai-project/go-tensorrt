
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

#include "test.hpp"

#include "imageNet.hpp"
// #include "cbits/util/loadImage.hpp"
// #include "imageNet.h"
// #include "loadImage.hpp"


#include "imageNet.cpp"
#include "tensorNet.cpp"
// #include "commandLine.cpp"
// #include "loadImage.cpp"




// main entry point
// int main( int argc, y )

int Start_code(float* imageData, const int width, const int height)
{

	
	const char* imgFilename = "./Orange.jpg";

	 char *a[2];
	a[0] = "blah";
	a[1] = "hmm";
	// create imageNet
	imageNet* net = imageNet::Create(1, a);

	if( !net )
	{
		printf("imagenet-console:   failed to initialize imageNet\n");
		return 0;
	}
	
	net->EnableProfiler();
	
	// load image from file on disk
	float* imgCPU    = NULL;
	float* imgCUDA   = NULL;
	int    imgWidth  = width;
	int    imgHeight = height;
		
	// if( !loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
	// {
	// 	printf("failed to load image '%s'\n", imgFilename);
	// 	return 0;
	// }

	// printf("loaded image  %s  (%u x %u)  %zu bytes\n", filename, imgWidth, imgHeight, imgSize);

	// allocate buffer for the image
	if( !cudaAllocMapped((void**)&imgCPU, (void**)&imgCUDA, imgWidth * imgHeight * sizeof(float) * 4) )
	for( int i = 0 ; i < imgWidth * imgHeight * 4; i ++) {
		imgCPU[i] = imageData[i];

	}
	float confidence = 0.0f;
	
	// classify image
	const int img_class = net->Classify(imgCUDA, imgWidth, imgHeight, &confidence);
	
	if( img_class >= 0 )
	{
		printf("imagenet-console:  '%s' -> %2.5f%% class #%i (%s)\n", imgFilename, confidence * 100.0f, img_class, net->GetClassDesc(img_class));
	
  }
	else
		printf("imagenet-console:  failed to classify '%s'  (result=%i)\n", imgFilename, img_class);
	
	printf("\nshutting down...\n");
	CUDA(cudaFreeHost(imgCPU));
	delete net;
	return 0;
}

