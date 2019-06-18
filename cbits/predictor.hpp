#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

    typedef void *PredictorHandle;

    PredictorHandle NewTensorRT(char *deploy_file,
                                char *weights_file,
                                int batch_size,
                                char **input_layer_name,
                                int len_input_layer_name,
                                char **output_layer_name,
                                int len_output_layer_name,
                                int shape_len);

    void InitTensorRT();

    void PredictTensorRT(PredictorHandle pred, float *imageData);

    float *GetPredictionsTensorRT(PredictorHandle pred);

    void DeleteTensorRT(PredictorHandle pred);

    void StartProfilingTensorRT(PredictorHandle pred, const char *name,
                                const char *metadata);

    void EndProfilingTensorRT(PredictorHandle pred);

    void DisableProfilingTensorRT(PredictorHandle pred);

    char *ReadProfileTensorRT(PredictorHandle pred);

    int GetShapeLenTensorRT(PredictorHandle pred);

    int GetPredLenTensorRT(PredictorHandle pred);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __PREDICTOR_HPP__
