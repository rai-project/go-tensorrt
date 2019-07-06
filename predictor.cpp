#ifdef __linux__
#include <algorithm>
#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include <NvInferPlugin.h>

#include "json.hpp"
#include "predictor.hpp"
#include "timer.h"
#include "timer.impl.hpp"

#include "half.hpp"

using namespace std;
using namespace nvinfer1;
using namespace nvinfer1;
using namespace nvcaffeparser1;
using std::string;

using json = nlohmann::json;

static bool has_error = false;
static std::string error_string{""};

static void clear_error() {
  has_error = false;
  error_string = "";
}

static void set_error(const std::string &err) {
  has_error = true;
  error_string = err;
}

#define START_C_DEFINION()                                                     \
  clear_error();                                                               \
  try {

#define END_C_DEFINION(res)                                                    \
  }                                                                            \
  catch (const std::exception &e) {                                            \
    std::cerr << "ERROR: " << e.what() << "\n";                                \
    set_error(e.what());                                                       \
  }                                                                            \
  catch (const std::string &e) {                                               \
    std::cerr << "ERROR: " << e << "\n";                                       \
    set_error(e);                                                              \
  }                                                                            \
  catch (...) {                                                                \
    std::cerr << "ERROR: unknown exception in go-tensorrt"                     \
              << "\n";                                                         \
    set_error("unknown exception in go-tensorrt");                             \
  }                                                                            \
  clear_error();                                                               \
  return res

class Logger : public ILogger {
  void log(Severity severity, const char *msg) override {
    // suppress info-level messages
    if (severity != Severity::kINFO) {
      std::cout << msg << std::endl;
    }
  }
} gLogger;

#define CHECK(stmt) stmt

#define CHECK_ERROR(stmt) stmt

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
  Predictor(IExecutionContext *context,
            std::vector<std::string> input_layer_names,
            std::vector<std::string> output_layer_names, int32_t batch_size)
      : context_(context), input_layer_names_(input_layer_names),
        output_layer_names_(output_layer_names), batch_size_(batch_size) {
    cudaStreamCreate(&stream_);
    const ICudaEngine &engine = context_->getEngine();
    data_.resize(engine.getNbBindings());
  };
  void Run() {
    if (context_ == nullptr) {
      throw std::runtime_error("tensorrt prediction error  null context_");
    }
    const ICudaEngine &engine = context_->getEngine();

    if (engine.getNbBindings() !=
        input_layer_names_.size() + output_layer_names_.size()) {
      throw std::runtime_error(std::string("tensorrt prediction error on ") +
                               std::to_string(__LINE__));
    }

    Profiler profiler(prof_);

    // Set the custom profiler.
    context_->setProfiler(&profiler);

    context_->enqueue(batch_size_, data_.data(), stream_, nullptr);
  }
  template <typename T>
  void AddInput(const std::string &name, const T *host_data,
                size_t num_elements) {
    void *gpu_data = nullptr;
    const ICudaEngine &engine = context_->getEngine();
    const auto idx = engine.getBindingIndex(name.c_str());
    if (idx == -1) {
      throw std::runtime_error(std::string("invalid input name ") + name);
    }
    const auto byte_count = batch_size_ * num_elements * sizeof(T);
    CHECK_ERROR(cudaMalloc(&gpu_data, byte_count));
    CHECK_ERROR(cudaMemcpyAsync(gpu_data, host_data, byte_count,
                                cudaMemcpyHostToDevice, stream_));
    data_[idx] = gpu_data;
  }

  template <typename T>
  void AddOutput(const std::string &name, size_t num_elements) {
    void *gpu_data = nullptr;
    const ICudaEngine &engine = context_->getEngine();
    const auto idx = engine.getBindingIndex(name.c_str());
    if (idx == -1) {
      throw std::runtime_error(std::string("invalid output name ") + name);
    }
    const auto byte_count = batch_size_ * num_elements * sizeof(T);
    CHECK_ERROR(cudaMalloc(&gpu_data, byte_count));
    data_[idx] = gpu_data;
  }

  void *GetOutputData(const std::string &name) {
    synchronize();

    const ICudaEngine &engine = context_->getEngine();
    const auto idx = engine.getBindingIndex(name.c_str());
    if (idx == -1) {
      throw std::runtime_error(std::string("invalid output name ") + name);
    }

    if (engine.bindingIsInput(idx)) {
      throw std::runtime_error(std::string("the layer name is not an output ") +
                               name);
    }

    const auto shape = GetOutputShape(name);
    auto element_byte_count = 1;
    const auto data_type = engine.getBindingDataType(idx);
    const size_t num_elements =
        std::accumulate(begin(shape), end(shape), 1, std::multiplies<size_t>());
    std::cout << "shape = " << shape[0] << "\n";
    switch (data_type) {
#define DISPATCH_GET_OUTPUT(DType, CType)                                      \
  case DType:                                                                  \
    element_byte_count = sizeof(CType);                                        \
    break;                                                                     \
    TensorRT_DType_Dispatch(DISPATCH_GET_OUTPUT)
#undef DISPATCH_GET_OUTPUT
    case DataType::kFLOAT:
      element_byte_count = sizeof(float);
      break;
    case DataType::kHALF:
      element_byte_count = sizeof(short);
      break;
    case DataType::kINT8:
      element_byte_count = sizeof(int8_t);
      break;
    case DataType::kINT32:
      element_byte_count = sizeof(int32_t);
      break;
    default:
      throw std::runtime_error("unexpected output type");
    }
    const auto byte_count = num_elements * element_byte_count;
    void *res_data = malloc(byte_count);
    std::cout << "byte_count = " << byte_count << "\n";
    CHECK(cudaMemcpy(res_data, data_[idx], byte_count, cudaMemcpyDeviceToHost));
    return res_data;
  }
  std::vector<int32_t> GetOutputShape(const std::string &name) {
    synchronize();

    const ICudaEngine &engine = context_->getEngine();
    const auto idx = engine.getBindingIndex(name.c_str());
    if (idx == -1) {
      throw std::runtime_error(std::string("invalid output name ") + name);
    }

    const auto dims = engine.getBindingDimensions(idx);
    const auto ndims = dims.nbDims;
    std::cout << __LINE__ << "  >>> "
              << "name = " << name << "\n";
    std::cout << __LINE__ << "  >>> "
              << "ndims = " << ndims << "\n";
    std::vector<int> res{};
    for (int ii = 0; ii < ndims; ii++) {
      std::cout << __LINE__ << "  >>> " << ii << "  ==  " << dims.d[ii] << "\n";
      res.emplace_back(dims.d[ii]);
    }
    std::cout << __LINE__ << "  >>> "
              << "res.size() = " << res.size() << "\n";
    return res;
  }

  void synchronize() { CHECK(cudaStreamSynchronize(stream_)); }
  ~Predictor() {
    for (auto data : data_) {
      cudaFree(data);
    }
    if (context_) {
      context_->destroy();
    }
    if (prof_) {
      prof_->reset();
      delete prof_;
      prof_ = nullptr;
    }
  }

  IExecutionContext *context_{nullptr};
  std::vector<string> input_layer_names_{nullptr};
  std::vector<string> output_layer_names_{nullptr};
  int32_t batch_size_{1};
  std::vector<void *> data_{nullptr};
  cudaStream_t stream_{0};
  profile *prof_{nullptr};
  bool profile_enabled_{false};
};

Predictor *get_predictor_from_handle(PredictorHandle predictor_handle) {
  auto predictor = (Predictor *)predictor_handle;
  if (predictor == nullptr) {
    throw std::runtime_error("expecting a non-nil predictor");
  }
  return predictor;
}

PredictorHandle
NewTensorRTPredictor(TensorRT_ModelFormat model_format, char *deploy_file,
                     char *weights_file, TensorRT_DType model_datatype,
                     char **input_layer_names, int32_t num_input_layer_names,
                     char **output_layer_names, int32_t num_output_layer_names,
                     int32_t batch_size) {

  START_C_DEFINION();

  // Create the builder
  IBuilder *builder = createInferBuilder(gLogger);
  if (builder == nullptr) {
    std::string err =
        std::string("cannot create tensorrt builder for ") + deploy_file;
    throw std::runtime_error(err);
  }

  // Parse the caffe model to populate the network, then set the outputs
  INetworkDefinition *network = builder->createNetwork();
  ICaffeParser *parser = createCaffeParser();
  assert(model_format == TensorRT_CaffeFormat);
  if (parser == nullptr) {
    std::string err =
        std::string("cannot create tensorrt paser for ") + deploy_file;
    throw std::runtime_error(err);
  }

  DataType blob_data_type = DataType::kFLOAT;
  switch (model_datatype) {
  case TensorRT_Byte:
    blob_data_type = DataType::kINT8;
    break;
  case TensorRT_Char:
    blob_data_type = DataType::kINT8;
    break;
  case TensorRT_Int:
    blob_data_type = DataType::kINT32;
    break;
  case TensorRT_Half:
    blob_data_type = DataType::kHALF;
    break;
  case TensorRT_Float:
    blob_data_type = DataType::kFLOAT;
    break;
  default:
    throw std::runtime_error("invalid model datatype");
  }
  const IBlobNameToTensor *blobNameToTensor =
      parser->parse(deploy_file, weights_file, *network, blob_data_type);

  std::vector<std::string> input_layer_names_vec{};
  for (int ii = 0; ii < num_input_layer_names; ii++) {
    input_layer_names_vec.emplace_back(input_layer_names[ii]);
  }
  std::vector<std::string> output_layer_names_vec{};
  for (int ii = 0; ii < num_output_layer_names; ii++) {
    output_layer_names_vec.emplace_back(output_layer_names[ii]);
    network->markOutput(*blobNameToTensor->find(output_layer_names[ii]));
  }

  builder->setMaxBatchSize(batch_size);
  builder->setMaxWorkspaceSize(36 << 20);
  builder->allowGPUFallback(true);

  builder->setInt8Mode(blob_data_type == DataType::kINT8);
  builder->setFp16Mode(blob_data_type == DataType::kHALF);

  ICudaEngine *engine = builder->buildCudaEngine(*network);

  network->destroy();
  parser->destroy();

  IHostMemory *trtModelStream = engine->serialize();

  engine->destroy();
  builder->destroy();

  IRuntime *runtime = createInferRuntime(gLogger);
  // Deserialize the engine
  ICudaEngine *runtime_engine = runtime->deserializeCudaEngine(
      trtModelStream->data(), trtModelStream->size(), nullptr);

  IExecutionContext *context = runtime_engine->createExecutionContext();

  trtModelStream->destroy();

  auto predictor = new Predictor(context, input_layer_names_vec,
                                 output_layer_names_vec, batch_size);

  return (PredictorHandle)predictor;

  END_C_DEFINION(nullptr);
}

void TenorRTPredictor_AddInput(PredictorHandle predictor_handle,
                               const char *name, TensorRT_DType dtype,
                               void *host_data, size_t num_elements) {
  START_C_DEFINION();
  auto predictor = get_predictor_from_handle(predictor_handle);
  switch (dtype) {
#define DISPATCH_ADD_INPUT(DType, CType)                                       \
  case DType:                                                                  \
    predictor->AddInput<CType>(name, reinterpret_cast<CType *>(host_data),     \
                               num_elements);                                  \
    break;
    TensorRT_DType_Dispatch(DISPATCH_ADD_INPUT);
#undef DISPATCH_ADD_INPUT
  default:
    throw std::runtime_error("unexpected input type");
  }
  END_C_DEFINION();
}

void TenorRTPredictor_Synchronize(PredictorHandle predictor_handle) {
  START_C_DEFINION();
  auto predictor = get_predictor_from_handle(predictor_handle);
  CHECK(predictor->synchronize());
  END_C_DEFINION();
}

void TenorRTPredictor_Run(PredictorHandle predictor_handle) {
  START_C_DEFINION();
  auto predictor = get_predictor_from_handle(predictor_handle);
  predictor->Run();
  END_C_DEFINION();
}

int TenorRTPredictor_GetNumOutputs(PredictorHandle predictor_handle) {
  START_C_DEFINION();
  auto predictor = get_predictor_from_handle(predictor_handle);
  return predictor->output_layer_names_.size();
  END_C_DEFINION(-1);
}

void *TenorRTPredictor_GetOutput(PredictorHandle predictor_handle,
                                 const char *name, int32_t *ndims,
                                 int32_t **res_dims) {
  START_C_DEFINION();
  auto predictor = get_predictor_from_handle(predictor_handle);

  std::cout << __LINE__ << "  >>> \n";
  auto dims = predictor->GetOutputShape(name);
  std::cout << __LINE__ << "  >>> \n";
  void *data = predictor->GetOutputData(name);
  *ndims = dims.size();

  std::cout << __LINE__ << "  >>> "
            << "*ndims = " << *ndims << "\n";

  *res_dims = (int32_t *)malloc(sizeof(int32_t) * (*ndims));
  memcpy(*res_dims, dims.data(), sizeof(int32_t) * (*ndims));
  return data;
  END_C_DEFINION(nullptr);
}

bool TenorRTPredictor_HasError(PredictorHandle predictor_handle) {
  return has_error;
}

const char *TenorRTPredictor_GetLastError(PredictorHandle predictor_handle) {
  return error_string.c_str();
}

void TenorRTPredictor_Delete(PredictorHandle predictor_handle) {
  START_C_DEFINION();
  auto predictor = get_predictor_from_handle(predictor_handle);
  if (predictor != nullptr) {
    delete predictor;
  }
  END_C_DEFINION();
}

void TenorRTPredictor_StartProfiling(PredictorHandle predictor_handle,
                                     const char *name, const char *metadata) {

  START_C_DEFINION();
  auto predictor = get_predictor_from_handle(predictor_handle);
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
  END_C_DEFINION();
}

void TenorRTPredictor_EndProfiling(PredictorHandle pred) {
  START_C_DEFINION();
  auto predictor = get_predictor_from_handle(pred);
  if (predictor->prof_) {
    predictor->prof_->end();
  }
  END_C_DEFINION();
}

char *TenorRTPredictor_ReadProfiling(PredictorHandle pred) {
  START_C_DEFINION();
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
  END_C_DEFINION(nullptr);
}

void TensoRT_Init() { initLibNvInferPlugins(&gLogger, ""); }

#endif // __linux__
