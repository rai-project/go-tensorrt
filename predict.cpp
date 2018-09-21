#ifdef __linux__

#include "predict.hpp"
#include "rapidjson-amalgamation.h"
#include "timer.h"
#include "timer.impl.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>

#include <cuda_runtime_api.h>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "dtoa_milo.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

#if 0
#define BACKWARD_HAS_DW 1
#include "backward.hpp"

namespace backward {

backward::SignalHandling sh;

} // namespace backward
#endif

class Logger : public ILogger {
  void log(Severity severity, const char *msg) override {
    // suppress info-level messages
    if (severity != Severity::kINFO) {
      std::cout << msg << std::endl;
    }
  }
} gLogger;

#if 0
#define CHECK(status)                                                          \
  {                                                                            \
    if (status != 0) {                                                         \
      std::cerr << "Cuda failure on line " << __LINE__                         \
                << " status =  " << status << "\n";                            \
      return nullptr;                                                          \
    }                                                                          \
  }
#else
#define CHECK(status) status;
#endif

class Profiler : public IProfiler {
public:
  Profiler(profile *prof) : prof_(prof) {
    if (prof_ == nullptr) {
      return;
    }
    current_time_ = prof_->get_start();
  }

  /** \brief layer time reporting callback
   *
   * \param layerName the name of the layer, set when constructing the network
   * definition
   * \param ms the time in milliseconds to execute the layer
   */
  virtual void reportLayerTime(const char *layer_name, float ms) {

    if (prof_ == nullptr) {
      return;
    }

    shapes_t shapes{};

    auto duration = std::chrono::nanoseconds((timestamp_t::rep)(1000000 * ms));
    auto e = new profile_entry(current_layer_sequence_index_, layer_name, "",
                               shapes);
#if 0
	timestamp_t n = now();
    e->set_start(n - duration);
    e->set_end(n);
    prof_->add(current_layer_sequence_index_ - 1, e);
#else
    e->set_start(current_time_);
    e->set_end(current_time_ + duration);
    prof_->add(current_layer_sequence_index_ - 1, e);
#endif

    current_layer_sequence_index_++;
    current_time_ += duration;
  }

  virtual ~Profiler() {}

private:
  profile *prof_{nullptr};
  int current_layer_sequence_index_{1};
  timestamp_t current_time_{};
};

class Predictor {
public:
  Predictor(ICudaEngine *engine, IExecutionContext *context)
      : engine_(engine), context_(context){};
  ~Predictor() {
    if (context_) {
      context_->destroy();
    }
    if (engine_) {
      engine_->destroy();
    }
    if (prof_) {
      prof_->reset();
      delete prof_;
      prof_ = nullptr;
    }
  }

  ICudaEngine *engine_{nullptr};
  IExecutionContext *context_{nullptr};
  profile *prof_{nullptr};
  bool prof_registered_{false};
};

PredictorContext NewTensorRT(char *deploy_file, char *weights_file, int batch,
                             char *outputLayer) {
  try {
    IBuilder *builder = createInferBuilder(gLogger);
    INetworkDefinition *network = builder->createNetwork();
    ICaffeParser *parser = createCaffeParser();

    const IBlobNameToTensor *blobNameToTensor =
        parser->parse(deploy_file, weights_file, *network, DataType::kFLOAT);

    auto loc = blobNameToTensor->find(outputLayer);
    if (loc == nullptr) {
      std::cerr << "cannot find " << outputLayer << " in blobNameToTensor\n";
      return nullptr;
    }
    network->markOutput(*loc);

    builder->setMaxBatchSize(batch);
    builder->setMaxWorkspaceSize(20 << 20);
    ICudaEngine *engine = builder->buildCudaEngine(*network);
    IExecutionContext *context = engine->createExecutionContext();
    Predictor *pred = new Predictor(engine, context);
    return (PredictorContext)pred;
  } catch (const std::invalid_argument &ex) {
    return nullptr;
  }
}

void DeleteTensorRT(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  delete predictor;
}

const char *PredictTensorRT(PredictorContext pred, float *input,
                            const char *input_layer_name,
                            const char *output_layer_name,
                            const int batchSize) {

  auto predictor = (Predictor *)pred;

  if (predictor == nullptr) {
    std::cerr << "tensorrt prediction error on " << __LINE__
              << " :: null predictor\n";
    return nullptr;
  }
  auto engine = predictor->engine_;
  if (engine->getNbBindings() != 2) {
    std::cerr << "tensorrt prediction error on " << __LINE__ << "\n";
    return nullptr;
  }
  auto context = predictor->context_;
  if (context == nullptr) {
    std::cerr << "tensorrt prediction error on " << __LINE__
              << " :: null context\n";
    return nullptr;
  }

  // In order to bind the buffers, we need to know the names of the input and
  // output tensors.
  // note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int input_index = engine->getBindingIndex(input_layer_name);
  const int output_index = engine->getBindingIndex(output_layer_name);

  // std::cerr << "using input layer = " << input_layer_name << "\n";
  // std::cerr << "using output layer = " << output_layer_name << "\n";

  const auto input_dim_ =
      static_cast<DimsCHW &&>(engine->getBindingDimensions(input_index));
  const auto input_byte_size =
      input_dim_.c() * input_dim_.h() * input_dim_.w() * sizeof(float);

  const auto output_dim_ =
      static_cast<DimsCHW &&>(engine->getBindingDimensions(output_index));
  const auto output_size = output_dim_.c() * output_dim_.h() * output_dim_.w();
  const auto output_byte_size = output_size * sizeof(float);

  float *input_layer, *output_layer;

  CHECK(cudaMalloc((void **)&input_layer, batchSize * input_byte_size));
  CHECK(cudaMalloc((void **)&output_layer, batchSize * output_byte_size));

  // std::cerr << "size of input = " << batchSize * input_byte_size << "\n";
  // std::cerr << "size of output = " << batchSize * output_byte_size << "\n";

  // DMA the input to the GPU,  execute the batch asynchronously, and DMA it
  // back:
  CHECK(cudaMemcpy(input_layer, input, batchSize * input_byte_size,
                   cudaMemcpyHostToDevice));

  void *buffers[2] = {input_layer, output_layer};

  Profiler profiler(predictor->prof_);

  // Set the custom profiler.
  context->setProfiler(&profiler);

  context->execute(batchSize, buffers);

  std::vector<float> output(batchSize * output_size);
  std::fill(output.begin(), output.end(), 0);

  CHECK(cudaMemcpy(output.data(), output_layer, batchSize * output_byte_size,
                   cudaMemcpyDeviceToHost));

  // release the stream and the buffers
  CHECK(cudaFree(input_layer));
  CHECK(cudaFree(output_layer));

#if 0
  // classify image
  rapidjson::Document preds;
preds.SetArray();

rapidjson::Document::AllocatorType& allocator = preds.GetAllocator();

const auto output_data = output.data();
  for (int cnt = 0; cnt < batchSize; cnt++) {
    for (int idx = 0; idx < output_size; idx++) {
		rapidjson::Value pred(rapidjson::kObjectType);
	pred.AddMember("index", idx, allocator);
	pred.AddMember("probability", output_data[cnt * output_size + idx], allocator);
	preds.PushBack(pred, allocator);
    }
  }


rapidjson::StringBuffer strbuf;
rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
preds.Accept(writer);


  return strbuf.GetString();
#else
  std::stringstream os;
  os << "[";
  for (int cnt = 0; cnt < batchSize; cnt++) {
    for (int idx = 0; idx < output_size; idx++) {
      if (cnt != 0 && idx != 0) {
        os << ",";
      }
      os << "{\"index\":" << idx << ", \"probability\":"
         << milo::dtoa_milo(output[cnt * output_size + idx]) << "}";
    }
  }
  os << "]";
  return  os.str().c_str();
#endif
}

void TensorRTInit() {}

void TensorRTStartProfiling(PredictorContext pred, const char *name,
                            const char *metadata) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (name == nullptr) {
    name = "";
  }
  if (metadata == nullptr) {
    metadata = "";
  }
  if (predictor->prof_ == nullptr) {
    predictor->prof_ = new profile(name, metadata);
  } else {
    predictor->prof_->reset();
  }
}

void TensorRTEndProfiling(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->end();
  }
}

void TensorRTDisableProfiling(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->reset();
  }
}

char *TensorRTReadProfile(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return strdup("");
  }
  if (predictor->prof_ == nullptr) {
    return strdup("");
  }
  const auto s = predictor->prof_->read();
  const auto cstr = s.c_str();
  return strdup(cstr);
}

#endif // __linux__
