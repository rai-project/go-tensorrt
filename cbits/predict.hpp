#ifndef __TEST_HPP__
#define __TEST_HPP__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef void *PredictorContext;
// int Start_code(int y);
// int Start_code(float* imageData, const int width, const int height);
PredictorContext NewTensorRT(char *model_file, char *trained_file, int batch, char* class_info);
void DeleteTensorRT();
const char *PredictTensorRT(PredictorContext pred, float *imageData,
                                const int width, const int height);
#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __TEST_HPP__