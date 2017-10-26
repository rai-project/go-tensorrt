#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stddef.h>

typedef void *PredictorContext;

PredictorContext CaffeNew(char *model_file, char *trained_file, int batch);

const char *CaffePredict(PredictorContext pred, float *imageData);

int CaffePredictorGetChannels(PredictorContext pred);

int CaffePredictorGetWidth(PredictorContext pred);

int CaffePredictorGetHeight(PredictorContext pred);

int CaffePredictorGetBatchSize(PredictorContext pred);

void CaffeDelete(PredictorContext pred);

void CaffeSetMode(int mode);

void CaffeInit();

void CaffeStartProfiling(PredictorContext pred, const char *name,
                         const char *metadata);

void CaffeEndProfiling(PredictorContext pred);

void CaffeDisableProfiling(PredictorContext pred);

char *CaffeReadProfile(PredictorContext pred);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __PREDICTOR_HPP__
