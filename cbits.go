// +build linux,!ppc64le

package tensorrt

// #include <stdlib.h>
// #include "cbits/predict.hpp"
import "C"
import (
	"encoding/json"
	"unsafe"

	"github.com/Unknwon/com"
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework/framework/options"
)

type Predictor struct {
	ctx     C.PredictorContext
	options *options.Options
}

func New(opts0 ...options.Option) (*Predictor, error) {
	opts := options.New(opts0...)
	modelFile := string(opts.Graph())
	if !com.IsFile(modelFile) {
		return nil, errors.Errorf("file %s not found", modelFile)
	}
	weightsFile := string(opts.Weights())
	if !com.IsFile(weightsFile) {
		return nil, errors.Errorf("file %s not found", weightsFile)
	}

	if opts.OutputNode() == "" {
		return nil, errors.Errorf("expecting a valid (non-empty) output node name")
	}

	modelFileString := C.CString(modelFile)
	defer C.free(unsafe.Pointer(modelFileString))

	weightsFileString := C.CString(weightsFile)
	defer C.free(unsafe.Pointer(weightsFileString))

	outputNodeString := C.CString(opts.OutputNode())
	defer C.free(unsafe.Pointer(outputNodeString))

	ctx := C.NewTensorRT(
		modelFileString,
		weightsFileString,
		C.int(opts.BatchSize()),
		outputNodeString,
	)
	return &Predictor{
		ctx:     ctx,
		options: opts,
	}, nil
}

func prod(arry []int32) int64 {
	accum := int64(1)
	for _, e := range arry {
		accum *= int64(e)
	}
	return accum
}

func (p *Predictor) Predict(inputLayerName0 string, outputLayerName0 string, input []float32, shape []int32) (Predictions, error) {
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

	batchSize := int64(p.options.BatchSize())
	shapeLen := prod(shape)
	dataLen := int64(len(input)) / shapeLen
	if batchSize > dataLen {
		padding := make([]float32, (batchSize-dataLen)*shapeLen)
		input = append(input, padding...)
	}

	ptr := (*C.float)(unsafe.Pointer(&input[0]))
	r := C.PredictTensorRT(p.ctx, ptr, inputLayerName, outputLayerName,
		C.int(p.options.BatchSize()),
	)
	if r == nil {
		return nil, errors.New("failed to perform tensorrt prediction")
	}
	defer C.free(unsafe.Pointer(r))
	js := C.GoString(r)

	predictions := []Prediction{}
	err := json.Unmarshal([]byte(js), &predictions)
	if err != nil {
		return nil, err
	}
	return predictions, nil
}
func (p *Predictor) StartProfiling(name, metadata string) error {
	cname := C.CString(name)
	cmetadata := C.CString(metadata)
	defer C.free(unsafe.Pointer(cname))
	defer C.free(unsafe.Pointer(cmetadata))
	C.TensorRTStartProfiling(p.ctx, cname, cmetadata)
	return nil
}

func (p *Predictor) EndProfiling() error {
	C.TensorRTEndProfiling(p.ctx)
	return nil
}

func (p *Predictor) DisableProfiling() error {
	C.TensorRTDisableProfiling(p.ctx)
	return nil
}

func (p *Predictor) ReadProfile() (string, error) {
	cstr := C.TensorRTReadProfile(p.ctx)
	if cstr == nil {
		return "", errors.New("failed to read nil profile")
	}
	defer C.free(unsafe.Pointer(cstr))
	return C.GoString(cstr), nil
}

func (p Predictor) Close() {
	C.DeleteTensorRT(p.ctx)
}
