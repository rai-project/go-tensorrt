#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#include "stdbool.h"
#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

  typedef void *PredictorHandle;

  // typedef int TensorRT_ModelFormat;

  // static const TensorRT_ModelFormat TensorRT_CaffeFormat = 1;
  // static const TensorRT_ModelFormat TensorRT_OnnxFormat = 2;
  // static const TensorRT_ModelFormat TensorRT_TensorFlowFormat = 3;

  typedef enum TensorRT_ModelFormat
  {
    TensorRT_CaffeFormat = 1,
    TensorRT_OnnxFormat = 2,
    TensorRT_UffFormat = 3,
  } TensorRT_ModelFormat;

  typedef enum TensorRT_DType
  {
    TensorRT_Unknown = 0,
    TensorRT_Byte = 1,
    TensorRT_Char = 2,
    TensorRT_Short = 3,
    TensorRT_Int = 4,
    TensorRT_Long = 5,
    TensorRT_Half = 6,
    TensorRT_Float = 7,
    TensorRT_Double = 8
  } TensorRT_DType;

  // typedef int TensorRT_DType;

  // static const TensorRT_DType TensorRT_Unknown = 0;
  // static const TensorRT_DType TensorRT_Byte = 1;
  // static const TensorRT_DType TensorRT_Char = 2;
  // static const TensorRT_DType TensorRT_Short = 3;
  // static const TensorRT_DType TensorRT_Int = 4;
  // static const TensorRT_DType TensorRT_Long = 5;
  // static const TensorRT_DType TensorRT_Half = 6;
  // static const TensorRT_DType TensorRT_Float = 7;
  // static const TensorRT_DType TensorRT_Double = 8;

#define TensorRT_DType_Dispatch(X) \
  X(TensorRT_Byte, uint8_t)        \
  X(TensorRT_Char, char)           \
  X(TensorRT_Short, int16_t)       \
  X(TensorRT_Int, int32_t)         \
  X(TensorRT_Long, int64_t)        \
  X(TensorRT_Half, float16)        \
  X(TensorRT_Float, float)         \
  X(TensorRT_Double, double)

  PredictorHandle NewTensorRTCaffePredictor(char *deploy_file,
                                            char *weights_file,
                                            TensorRT_DType model_datatype,
                                            char **input_layer_names, 
                                            int32_t num_input_layer_names,
                                            char **output_layer_names, 
                                            int32_t num_output_layer_names,
                                            int32_t batch_size);

  PredictorHandle NewTensorRTUffPredictor(char *model_file, 
                                          TensorRT_DType model_datatype,
                                          int **input_shapes,
                                          char **input_orders,
                                          char **input_layer_names, 
                                          int32_t num_input_layer_names,
                                          char **output_layer_names, 
                                          int32_t num_output_layer_names,
                                          int32_t batch_size);

  PredictorHandle NewTensorRTOnnxPredictor(char *model_file, 
                                           TensorRT_DType model_datatype,
                                           char **input_layer_names, 
                                           int32_t num_input_layer_names,
                                           char **output_layer_names, 
                                           int32_t num_output_layer_names,
                                           int32_t batch_size);

  void TensorRTPredictor_SetDevice(PredictorHandle pred, int32_t device);

  void TensorRTPredictor_AddInput(PredictorHandle pred, const char *name,
                                 TensorRT_DType dtype, void *data,
                                 size_t num_elements);

  void TensorRTPredictor_AddOutput(PredictorHandle pred, const char *name,
                                  TensorRT_DType dtype);

  void TensorRTPredictor_Synchronize(PredictorHandle pred);

  void TensorRTPredictor_Run(PredictorHandle pred);

  int TensorRTPredictor_GetNumOutputs(PredictorHandle pred);

  void *TensorRTPredictor_GetOutput(PredictorHandle pred, const char *name,
                                   int32_t *ndims, int32_t **dims);

  bool TensorRTPredictor_HasError(PredictorHandle pred);

  const char *TensorRTPredictor_GetLastError(PredictorHandle pred);

  void TensorRTPredictor_Delete(PredictorHandle pred);

  void TensorRTPredictor_StartProfiling(PredictorHandle pred, const char *name,
                                       const char *metadata);

  void TensorRTPredictor_EndProfiling(PredictorHandle pred);

  char *TensorRTPredictor_ReadProfiling(PredictorHandle pred);

  void TensoRT_Init();

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __PREDICTOR_HPP__
