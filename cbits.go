// +build linux,!ppc64le

package tensorrt

// #include <stdlib.h>
// #include "cbits/predictor.hpp"
import "C"
import (
	"context"
	"unsafe"

	"github.com/Unknwon/com"
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework/framework/options"
	"github.com/rai-project/tracer"
)

type Predictor struct {
	ctx     C.PredictorContext
	options *options.Options
}

func New(ctx context.Context, opts ...options.Option) (*Predictor, error) {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_new")
	defer span.Finish()

	options := options.New(opts...)
	modelFile := string(options.Graph())
	if !com.IsFile(modelFile) {
		return nil, errors.Errorf("file %s not found", modelFile)
	}
	weightsFile := string(options.Weights())
	if !com.IsFile(weightsFile) {
		return nil, errors.Errorf("file %s not found", weightsFile)
	}

	if options.OutputNode() == "" {
		return nil, errors.Errorf("expecting a valid (non-empty) output node name")
	}

	modelFileString := C.CString(modelFile)
	defer C.free(unsafe.Pointer(modelFileString))

	weightsFileString := C.CString(weightsFile)
	defer C.free(unsafe.Pointer(weightsFileString))

	outputNodeString := C.CString(options.OutputNode())
	defer C.free(unsafe.Pointer(outputNodeString))

	return &Predictor{
		ctx: C.NewTensorRT(
			modelFileString,
			weightsFileString,
			C.int(options.BatchSize()),
			outputNodeString,
		),
		options: options,
	}, nil
}

func prod(arry []uint32) int64 {
	accum := int64(1)
	for _, e := range arry {
		accum *= int64(e)
	}
	return accum
}

func (p *Predictor) Predict(inputLayerName0 string, outputLayerName0 string, input []float32, shape []uint32) (Predictions, error) {
	// log.WithField("input_layer_name", inputLayerName0).
	// 	WithField("output_layer_name", outputLayerName0).
	// 	Info("performing tensorrt prediction")

	if inputLayerName0 == "" {
		return nil, errors.New("expecting a valid (non-empty) input layer name")
	}

	if outputLayerName0 == "" {
		return nil, errors.New("expecting a valid (non-empty) output layer name")
	}

	inputLayerName := C.CString(inputLayerName0)
	defer C.free(unsafe.Pointer(inputLayerName))

	outputLayerName := C.CString(outputLayerName0)
	defer C.free(unsafe.Pointer(outputLayerName))

	batchSize := p.options.BatchSize()
	shapeLen := prod(shape)
	dataLen := len(data)

	inputCount := dataLen / shapeLen
	if batchSize > inputCount {
		padding := make([]float32, (batchSize-inputCount)*shapeLen)
		data = append(data, padding...)
	}

	ptr := (*C.float)(unsafe.Pointer(&input[0]))

	predictSpan, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_predict")
	defer predictSpan.Finish()

	C.PredictTensorRT(p.ctx, ptr, inputLayerName, outputLayerName, C.int(batchSize))

	return nil
}

func (p *Predictor) ReadPredictedFeatures(ctx context.Context) Predictions {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "read_predicted_features")
	defer span.Finish()

	batchSize := p.options.BatchSize()
	predLen := int(C.GetPredLenCaffe(p.ctx))
	length := batchSize * predLen

	cPredictions := C.GetPredictionsCaffe(p.ctx)

	slice := (*[1 << 30]C.float)(unsafe.Pointer(cPredictions))[:length:length]

	predictions := make([]Prediction, length)
	for ii := 0; ii < length; ii++ {
		predictions[ii] = Prediction{
			Index:       ii % predLen,
			Probability: float32(slice[ii]),
		}
	}

	return predictions
}

func (p Predictor) Close() {
	C.DeleteTensorRT(p.ctx)
}

func (p *Predictor) StartProfiling(name, metadata string) error {
	cname := C.CString(name)
	cmetadata := C.CString(metadata)
	defer C.free(unsafe.Pointer(cname))
	defer C.free(unsafe.Pointer(cmetadata))
	C.StartProfilingTensorRT(p.ctx, cname, cmetadata)
	return nil
}

func (p *Predictor) EndProfiling() error {
	C.EndProfilingTensorRT(p.ctx)
	return nil
}

func (p *Predictor) DisableProfiling() error {
	C.DisableProfilingTensorRT(p.ctx)
	return nil
}

func (p *Predictor) ReadProfile() (string, error) {
	cstr := C.ReadProfileTensorRT(p.ctx)
	if cstr == nil {
		return "", errors.New("failed to read nil profile")
	}
	defer C.free(unsafe.Pointer(cstr))
	return C.GoString(cstr), nil
}
