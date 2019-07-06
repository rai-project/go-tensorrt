#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef void *PredictorHandle;

typedef int TensorRT_ModelFormat;

static const TensorRT_ModelFormat TensorRT_CaffeFormat = 1;
static const TensorRT_ModelFormat TensorRT_OnnxFormat = 2;
static const TensorRT_ModelFormat TensorRT_TensorFlowFormat = 3;

typedef enum TensorRT_DType {
  TensorRT_Unknown = 0,
  TensorRT_Byte = 1,
  TensorRT_Char = 2,
  TensorRT_Short = 3,
  TensorRT_Int = 4,
  TensorRT_Long = 5,
  TensorRT_Half = 6,
  TensorRT_Float = 7,
  TensorRT_Double = 8,
} TensorRT_DType;

#define TensorRT_DType_Dispatch(X)                                             \
  X(TensorRT_Byte, uint8_t)                                                    \
  X(TensorRT_Char, char)                                                       \
  X(TensorRT_Short, int16_t)                                                   \
  X(TensorRT_Int, int32_t)                                                     \
  X(TensorRT_Long, int64_t)                                                    \
  X(TensorRT_Half, half)                                                       \
  X(TensorRT_Float, float)                                                     \
  X(TensorRT_Double, double)

PredictorHandle
NewTensorRTPredictor(TensorRT_ModelFormat model_format, char *deploy_file,
                     char *weights_file, TensorRT_DType model_datatype,
                     char **input_layer_names, int32_t num_input_layer_names,
                     char **output_layer_names, int32_t num_output_layer_names,
                     int32_t batch_size);

void TenorRTPredictor_SetDevice(PredictorHandle pred, int32_t device);

void TenorRTPredictor_AddInput(PredictorHandle pred, TensorRT_DType dtype,
                               void *data, size_t num_elements);

void TenorRTPredictor_Synchronize(PredictorHandle pred);

void TenorRTPredictor_Run(PredictorHandle pred);

void TenorRTPredictor_GetNumOutputs(PredictorHandle pred);

void *TenorRTPredictor_GetOutput(PredictorHandle pred, int32_t idx);

bool TenorRTPredictor_HasError(PredictorHandle pred);

const char *TenorRTPredictor_GetLastError(PredictorHandle pred);

float *GetPredictionsTensorRT(PredictorHandle pred);

void TenorRTPredictor_Delete(PredictorHandle pred);

void TenorRTPredictor_StartProfiling(PredictorHandle pred, const char *name,
                                     const char *metadata);

void TenorRTPredictor_EnableProfiling(PredictorHandle pred);

void TenorRTPredictor_DisableProfiling(PredictorHandle pred);

char *TenorRTPredictor_ReadProfiling(PredictorHandle pred);

void InitTensorRT();

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __PREDICTOR_HPP__
