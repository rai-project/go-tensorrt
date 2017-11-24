
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

#include "predict.hpp"
#include "json.hpp"

#include "imageNet.hpp"
#include "imageNet.cpp"
#include "tensorNet.cpp"
#include <iostream>
using json = nlohmann::json;

struct PredictorObject {
  PredictorObject(imageNet *ctx) : ctx_(ctx){};
  ~PredictorObject() {
    if (ctx_) {
      delete ctx_;
    }
  }
  imageNet *ctx_;

};

PredictorContext NewTensorRT(char *model_file, char *trained_file, int batch, char* class_info) {
  try
  {
    const auto ctx = imageNet::Create(model_file, trained_file, NULL, class_info);
		return (void *)ctx;
  }
  catch (const std::invalid_argument &ex)
  {
    return nullptr;
	}

}

void DeleteTensorRT(PredictorContext pred) {
  auto predictor = (imageNet *)pred;
  delete predictor;
}

const char *PredictTensorRT(PredictorContext pred, float *imageData, const int imgWidth, const int imgHeight) {

	float* imgCPU    = NULL;
	float* imgCUDA   = NULL;
	int imageSize = imgWidth * imgHeight * sizeof(float) * 4;

	cudaAllocMapped((void**)&imgCPU, (void**)&imgCUDA, imageSize);

memcpy(imgCPU, imageData, imageSize);

	float confidence = 0.0f;

	imageNet * net = (imageNet *)pred;
	
	// classify image
	const int img_class = net->Classify(imgCPU, imgWidth, imgHeight, &confidence);
  	json preds = json::array();
    preds.push_back(
       	{{"index", net->GetClassDesc(img_class)}, {"probability", confidence}});
  	auto res = strdup(preds.dump().c_str());
  	return res;
  }
