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

using json = nlohmann::json;

class Logger : public ILogger
{
  void log(Severity severity, const char *msg) override
  {
    // suppress info-level messages
    if (severity != Severity::kINFO)
    {
      std::cout << msg << std::endl;
    }
  }
} gLogger;

#define CHECK(status)                                  \
  {                                                    \
    if (status != 0)                                   \
    {                                                  \
      std::cerr << "Cuda failure on line " << __LINE__ \
                << " status =  " << status << "\n";    \
    }                                                  \
  }

class Profiler : public IProfiler
{
public:
  Profiler(profile *prof) : prof_(prof)
  {
    if (prof_ == nullptr)
    {
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
  virtual void reportLayerTime(const char *layer_name, float ms)
  {

    if (prof_ == nullptr)
    {
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

class Predictor
{
public:
  Predictor(ICudaEngine *engine, IExecutionContext *context, int batch_size,
            std::string input_layer_name, std::string output_layer_name, int shape_len)
      : engine_(engine), context_(context), batch_(batch_size),
        input_layer_name_(input_layer_name),
        output_layer_name_(output_layer_name), shape_len_(shape_len){};

  void Predict(float *imageData);

  ~Predictor()
  {
    if (context_)
    {
      context_->destroy();
    }
    if (engine_)
    {
      engine_->destroy();
    }
    if (prof_)
    {
      prof_->reset();
      delete prof_;
      prof_ = nullptr;
    }
  }

  ICudaEngine *engine_{nullptr};
  IExecutionContext *context_{nullptr};
  int batch_;
  std::string input_layer_name_{nullptr};
  std::string output_layer_name_{nullptr};
  int shape_len_;
  int pred_len_;
  float *result_{nullptr};
  profile *prof_{nullptr};
  bool profile_enabled_{false};
};

void Predictor::Predict(float *inputData)
{
  if (engine_->getNbBindings() != 2)
  {
    std::cerr << "tensorrt prediction error on " << __LINE__ << "\n";
  }
  if (context_ == nullptr)
  {
    std::cerr << "tensorrt prediction error on " << __LINE__
              << " :: null context_\n";
  }
  if (result_ != nullptr) {
      free(result_);
      result_ = nullptr;
  }

  // In order to bind the buffers, we need to know the names of the input and
  // output tensors.
  // note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int input_index = engine_->getBindingIndex(input_layer_name_.c_str());
  const int output_index = engine_->getBindingIndex(output_layer_name_.c_str());

  const auto input_dim_ =
      static_cast<DimsCHW &&>(engine_->getBindingDimensions(input_index));
  const auto input_byte_size =
      batch_ * input_dim_.c() * input_dim_.h() * input_dim_.w() * sizeof(float);

  const auto output_dim_ =
      static_cast<DimsCHW &&>(engine_->getBindingDimensions(output_index));
  pred_len_ = output_dim_.c() * output_dim_.h() * output_dim_.w();
  const auto output_byte_size = batch_ * pred_len_ * sizeof(float);

  float *input_layer, *output_layer;

  CHECK(cudaMalloc((void **)&input_layer, input_byte_size));
  CHECK(cudaMalloc((void **)&output_layer, output_byte_size));

  // std::cerr << "size of input = " <<  input_byte_size << "\n";
  // std::cerr << "size of output = " << output_byte_size << "\n";

  // DMA the input to the GPU,  execute the batch_size  asynchronously, and DMA
  // it back:
  CHECK(cudaMemcpy(input_layer, inputData, input_byte_size,
                   cudaMemcpyHostToDevice));

  void *buffers[2] = {input_layer, output_layer};

  Profiler profiler(prof_);

  // Set the custom profiler.
  context_->setProfiler(&profiler);

  context_->execute(batch_, buffers);

  result_ = (float *)malloc(output_byte_size);
  CHECK(cudaMemcpy(
      result_, output_layer, output_byte_size, cudaMemcpyDeviceToHost));

  // release the stream and the buffers
  CHECK(cudaFree(input_layer));
  CHECK(cudaFree(output_layer));
}

PredictorContext NewTensorRT(char *deploy_file, char *weights_file,
                             int batch_size, char *input_layer_name,
                             char *output_layer_name, int shape_len)
{
  try
  {
    IBuilder *builder = createInferBuilder(gLogger);
    INetworkDefinition *network = builder->createNetwork();
    ICaffeParser *parser = createCaffeParser();

    const IBlobNameToTensor *blobNameToTensor =
        parser->parse(deploy_file, weights_file, *network, DataType::kFLOAT);

    auto loc = blobNameToTensor->find(output_layer_name);
    if (loc == nullptr)
    {
      std::cerr << "cannot find " << output_layer_name
                << " in blobNameToTensor\n";
      return nullptr;
    }
    network->markOutput(*loc);

    builder->setMaxBatchSize(batch_size);
    builder->setMaxWorkspaceSize(20 << 20);
    ICudaEngine *engine = builder->buildCudaEngine(*network);
    IExecutionContext *context = engine->createExecutionContext();
    Predictor *pred =
        new Predictor(engine, context, batch_size, input_layer_name,
                      output_layer_name, shape_len);
    return (PredictorContext)pred;
  }
  catch (const std::invalid_argument &ex)
  {
    return nullptr;
  }
}

void InitTensorRT() {}

void PredictTensorRT(PredictorContext pred, float *inputData)
{
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr)
  {
    return;
  }
  predictor->Predict(inputData);
  return;
}

float *GetPredictionsTensorRT(PredictorContext pred)
{
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr)
  {
    return nullptr;
  }
  return predictor->result_;
}

void DeleteTensorRT(PredictorContext pred)
{
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr)
  {
    return;
  }

  if (predictor->result_)
  {
    free(predictor->result_);
    predictor->result_ = nullptr;
  }
  delete predictor;
}

void StartProfilingTensorRT(PredictorContext pred, const char *name,
                            const char *metadata)
{
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr)
  {
    return;
  }
  if (name == nullptr)
  {
    name = "";
  }
  if (metadata == nullptr)
  {
    metadata = "";
  }
  if (predictor->prof_ == nullptr)
  {
    predictor->prof_ = new profile(name, metadata);
  }
  else
  {
    predictor->prof_->reset();
  }
}

void EndProfilingTensorRT(PredictorContext pred)
{
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr)
  {
    return;
  }
  if (predictor->prof_)
  {
    predictor->prof_->end();
  }
}

void DisableProfilingTensorRT(PredictorContext pred)
{
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr)
  {
    return;
  }
  if (predictor->prof_)
  {
    predictor->prof_->reset();
  }
}

char *ReadProfileTensorRT(PredictorContext pred)
{
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr)
  {
    return strdup("");
  }
  if (predictor->prof_ == nullptr)
  {
    return strdup("");
  }
  const auto s = predictor->prof_->read();
  const auto cstr = s.c_str();
  return strdup(cstr);
}

int GetShapeLenTensorRT(PredictorContext pred)
{
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr)
  {
    return 0;
  }
  return predictor->shape_len_;
}

int GetPredLenTensorRT(PredictorContext pred)
{
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr)
  {
    return 0;
  }
  return predictor->pred_len_;
}

#endif // __linux__
