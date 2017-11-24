
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

#include <algorithm>
#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

#include "NvCaffeParser.h"
#include "NvInfer.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

using json = nlohmann::json;

class Logger : public ILogger {
  void log(Severity severity, const char *msg) override {
    // suppress info-level messages
    if (severity != Severity::kINFO) {
      std::cout << msg << std::endl;
    }
  }
} gLogger;

#define CHECK(status)                                                          \
  {                                                                            \
    if (status != 0) {                                                         \
      std::cout << "Cuda failure on line " << __LINE__                         \
                << " status =  " << status << "\n";                            \
      return nullptr;                                                          \
    }                                                                          \
  }

PredictorContext NewTensorRT(char *model_file, char *trained_file, int batch,
                             char *outputLayer) {
  try {
    IBuilder *builder = createInferBuilder(gLogger);
    INetworkDefinition *network = builder->createNetwork();
    ICaffeParser *parser = createCaffeParser();

    const IBlobNameToTensor *blobNameToTensor =
        parser->parse(model_file, trained_file, *network, DataType::kFLOAT);

    auto loc = blobNameToTensor->find(outputLayer);
    if (loc == nullptr) {
      std::cout << "cannot find " << outputLayer << " in blobNameToTensor\n";
      return nullptr;
    }
    network->markOutput(*loc);

    builder->setMaxBatchSize(batch);
    builder->setMaxWorkspaceSize(1 << 20);
    ICudaEngine *engine = builder->buildCudaEngine(*network);
    return (PredictorContext)engine;
  } catch (const std::invalid_argument &ex) {
    return nullptr;
  }
}

void DeleteTensorRT(PredictorContext pred) {
  auto predictor = (ICudaEngine *)pred;
  if (predictor == nullptr) {
    return;
  }
  predictor->destroy();
}

const char *PredictTensorRT(PredictorContext pred, float *input,
                            const char *input_layer_name,
                            const char *output_layer_name, const int batchSize) {

  auto predictor = (ICudaEngine *)pred;

  if (predictor == nullptr) {
    std::cout << "error on " << __LINE__ << "\n";
    return nullptr;
  }
  if (predictor->getNbBindings() != 2) {
    std::cout << "error on " << __LINE__ << "\n";
    return nullptr;
  }

  // In order to bind the buffers, we need to know the names of the input and
  // output tensors.
  // note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int input_index = predictor->getBindingIndex(input_layer_name);
  const int output_index = predictor->getBindingIndex(output_layer_name);

  std::cerr << "using input layer = " << input_layer_name << "\n";
  std::cerr << "using output layer = " << output_layer_name << "\n";

  const auto input_dim_ =
      static_cast<DimsCHW &&>(predictor->getBindingDimensions(input_index));
  const auto input_byte_size =
      input_dim_.c() * input_dim_.h() * input_dim_.w() * sizeof(float);

  const auto output_dim_ =
      static_cast<DimsCHW &&>(predictor->getBindingDimensions(output_index));
const auto output_size = output_dim_.c() * output_dim_.h() * output_dim_.w();
  const auto output_byte_size = output_size * sizeof(float);

  std::vector<float> output(batchSize * output_size);
  std::fill(output.begin(), output.end(), 0);

float * input_layer, * output_layer;

  CHECK(cudaMalloc((void**)&input_layer, batchSize * input_byte_size));
  CHECK(cudaMalloc((void**)&output_layer, batchSize * output_byte_size));

  IExecutionContext *context = predictor->createExecutionContext();

  std::cerr << "size of input = "
            << batchSize * input_byte_size
            << "\n";

  std::cerr << "size of output = " << batchSize * output_byte_size
            << "\n";

            for (int ii = 102 ; ii < 112; ii++) {
              std::cerr << "cinput [" << ii << " ] = " << input[ii] << "\n";
            }

  // DMA the input to the GPU,  execute the batch asynchronously, and DMA it
  // back:
  CHECK(cudaMemcpy(input_layer, input,
                   batchSize * input_byte_size,
                   cudaMemcpyHostToDevice));

void* buffers[2] = { input_layer, output_layer };
  context->execute(batchSize, buffers);
  CHECK(cudaMemcpy(output.data(), output_layer,
                   batchSize * output_byte_size,
                   cudaMemcpyDeviceToHost));

  // release the stream and the buffers
  CHECK(cudaFree(input_layer));
  CHECK(cudaFree(output_layer));

  context->destroy();

  // classify image
  json preds = json::array();

  for (int cnt = 0; cnt < batchSize; cnt++) {
    for (int idx = 0; idx < output_size; idx++) {
      preds.push_back(
          {{"index", idx}, {"probability", output[cnt * output_size + idx]}});
    }
  }

  auto res = strdup(preds.dump().c_str());
  return res;
}
