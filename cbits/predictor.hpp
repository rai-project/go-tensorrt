#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef void *PredictorContext;

PredictorContext NewTensorRT(char *deploy_file, char *weights_file,
                             int batch_size, char *input_layer_name,
                             char *output_layer_name, int shape_len);

void InitTensorRT();

void PredictTensorRT(PredictorContext pred, float *imageData);

const float *GetPredictionsTensorRT(PredictorContext pred);

void DeleteTensorRT(PredictorContext pred);

void StartProfilingTensorRT(PredictorContext pred, const char *name,
                            const char *metadata);

void EndProfilingTensorRT(PredictorContext pred);

void DisableProfilingTensorRT(PredictorContext pred);

char *ReadProfileTensorRT(PredictorContext pred);

int GetShapeLenTensorRT(PredictorContext pred);

int GetPredLenTensorRT(PredictorContext pred);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __PREDICTOR_HPP__
