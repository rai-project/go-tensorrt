#ifdef __linux__
#include <algorithm>
#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

#include "NvCaffeParser.h"
#include "NvInfer.h"

#include "json.hpp"
#include "predictor.hpp"
#include "timer.h"
#include "timer.impl.hpp"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using std::string;

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
      std::cerr << "cuda failure on line " << __LINE__                         \
                << " status =  " << status << "\n";                            \
    }                                                                          \
  }

class Profiler : public IProfiler {
public:
  Profiler(profile *prof) : prof_(prof) {
    if (prof_ == nullptr) {
      return;
    }
    prof_->start(); // reset start time
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
    e->set_start(current_time_);
    e->set_end(current_time_ + duration);
    prof_->add(current_layer_sequence_index_ - 1, e);

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
  Predictor(ICudaEngine *engine, IExecutionContext *context, int batch_size,
            std::vector<std::string> input_layer_names,
            std::vector<std::string> output_layer_names)
      : engine_(engine), context_(context), batch_(batch_size),
        input_layer_names_(input_layer_names),
        output_layer_names_(output_layer_names)){};
  void Predict();
  void SetInput(int idx, float *data);
  const float *GetOutputData(int idx);
  std::vector<int> GetOutputShape(int idx);

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
  int batch_;
  std::vector<string> input_layer_names_{nullptr};
  std::vector<string> output_layer_names_{nullptr};
  std::vector<float *> outputs_{nullptr};
  profile *prof_{nullptr};
  bool profile_enabled_{false};
};

void Predictor::SetInput() {}

void Predictor::Predict() {
  if (engine_->getNbBindings() != 2) {
    std::cerr << "tensorrt prediction error on " << __LINE__ << "\n";
  }
  if (context_ == nullptr) {
    std::cerr << "tensorrt prediction error on " << __LINE__
              << " :: null context_\n";
  }
  if (outputs_ != nullptr) {
    for (int ii = 0; ii < outputs_.size(); ii++) {
      free(outputs_[ii]);
    }
    outputs_ = nullptr;
  }

  // In order to bind the buffers, we need to know the names of the input and
  // output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int input_index =
      engine_->getBindingIndex(input_layer_names_[0].c_str());
  const auto input_dim_ =
      static_cast<DimsCHW &&>(engine_->getBindingDimensions(input_index));
  const auto input_byte_size =
      batch_ * input_dim_.c() * input_dim_.h() * input_dim_.w() * sizeof(float);

  const int output_index =
      engine_->getBindingIndex(output_layer_names_[0].c_str());
  const auto output_dim_ =
      static_cast<DimsCHW &&>(engine_->getBindingDimensions(output_index));
  pred_len_ = output_dim_.c() * output_dim_.h() * output_dim_.w();
  const auto output_byte_size = batch_ * pred_len_ * sizeof(float);

  float *input_layer, *output_layer;

  CHECK(cudaMalloc((void **)&input_layer, input_byte_size));
  CHECK(cudaMalloc((void **)&output_layer, output_byte_size));

  // std::cerr << "size of input = " <<  input_byte_size << "\n";
  // std::cerr << "size of output = " << output_byte_size << "\n";

  // DMA the input to the GPU,  execute the batch_size asynchronously, and DMA
  // it back:
  CHECK(cudaMemcpy(input_layer, inputData, input_byte_size,
                   cudaMemcpyHostToDevice));

  void *buffers[2] = {input_layer, output_layer};

  Profiler profiler(prof_);

  // Set the custom profiler.
  context_->setProfiler(&profiler);

  context_->execute(batch_, buffers);

  result_ = (float *)malloc(output_byte_size);
  CHECK(cudaMemcpy(result_, output_layer, output_byte_size,
                   cudaMemcpyDeviceToHost));

  // release the stream and the buffers
  CHECK(cudaFree(input_layer));
  CHECK(cudaFree(output_layer));
}

PredictorHandle
NewTensorRT(char *prototxt_file, // caffe prototxt
            char *weights_file,  // caffe model weights
            int batch_size, // batch size - NB must be at least as large as the
                            // batch we want to run with)
            char **input_layer_name, int len_input_layer_name,
            char **output_layer_name, int len_output_layer_name) {
  try {
    // Create the builder
    IBuilder *builder = createInferBuilder(gLogger);
    assert(builder != nullptr);

    // Parse the caffe model to populate the network, then set the outputs
    INetworkDefinition *network = builder->createNetwork();
    ICaffeParser *parser = createCaffeParser();

    const IBlobNameToTensor *blobNameToTensor =
        parser->parse(prototxt_file, weights_file, *network, DataType::kFLOAT);

    std::vector<std::string> input_layer_name_vec{};
    for (int ii = 0; ii < len_input_layer_name; ii++) {
      input_layer_name_vec.emplace_back(input_layer_name[ii]);
    }
    std::vector<std::string> output_layer_name_vec{};
    for (int ii = 0; ii < len_output_layer_name; ii++) {
      output_layer_name_vec.emplace_back(output_layer_name[ii]);
    }

    // Specify which tensors are outputs
    for (auto &s : output_layer_name_vec) {
      auto loc = blobNameToTensor->find(output_layer_name[0]);
      if (loc == nullptr) {
        std::cerr << "cannot find " << output_layer_name
                  << " in blobNameToTensor\n";
        return nullptr;
      }
      network->markOutput(*loc);
    }

    // Build the engine
    builder->setMaxBatchSize(batch_size);
    builder->setMaxWorkspaceSize(
        10 << 20); // We need about 6MB of scratch space for the plugin layer
                   // for batch size 5

    enableDLA(builder, gArgs.useDLACore);

    ICudaEngine *engine = builder->buildCudaEngine(*network);
    assert(engine);

    // We don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    IExecutionContext *context = engine->createExecutionContext();
    Predictor *pred =
        new Predictor(engine, context, batch_size, input_layer_name_vec,
                      output_layer_name_vec);
    return (PredictorHandle)pred;
  } catch (const std::invalid_argument &ex) {
    return nullptr;
  }
}

void PredictTensorRT(PredictorHandle pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  predictor->Predict();
  return;
}

float *GetPredictionsTensorRT(PredictorHandle pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return nullptr;
  }
  return predictor->result_;
}

void DeleteTensorRT(PredictorHandle pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }

  if (predictor->result_) {
    free(predictor->result_);
    predictor->result_ = nullptr;
  }
  delete predictor;
}

void StartProfilingTensorRT(PredictorHandle pred, const char *name,
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

void EndProfilingTensorRT(PredictorHandle pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->end();
  }
}

void DisableProfilingTensorRT(PredictorHandle pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->reset();
  }
}

char *ReadProfileTensorRT(PredictorHandle pred) {
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

const int *GetOutputShapeTensorRT(PredictorHandle pred, int32_t idx,
                                  int32_t *len) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return nullptr;
  }
  const auto shape = predictor->GetOutputShape(idx);
  *len = shape.size();
  const auto byte_count = shape.size() * sizeof(int);
  int *res = (int *)malloc(byte_count);
  memcpy(res, shape.data(), byte_count);
  return res;
}

int GetPredLenTensorRT(PredictorHandle pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  return predictor->pred_len_;
}

#endif // __linux__

inline void enableDLA(IBuilder *b, int useDLACore,
                      bool allowGPUFallback = true) {
  if (useDLACore >= 0) {
    if (b->getNbDLACores() == 0) {
      std::cerr << "Trying to use DLA core " << useDLACore
                << " on a platform that doesn't have any DLA cores"
                << std::endl;
      assert(
          "Error: use DLA core on a platfrom that doesn't have any DLA cores" &&
          false);
    }
    b->allowGPUFallback(allowGPUFallback);
    if (!b->getInt8Mode()) {
      // User has not requested INT8 Mode.
      // By default run in FP16 mode. FP32 mode is not permitted.
      b->setFp16Mode(true);
    }
    b->setDefaultDeviceType(DeviceType::kDLA);
    b->setDLACore(useDLACore);
    b->setStrictTypeConstraints(true);
  }
}
