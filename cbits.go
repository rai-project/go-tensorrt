// +build linux

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

func (p *Predictor) Predict(input []float32) (Predictions, error) {
	inputLayerName := C.CString(p.options.InputNodes()[0].Key())
	defer C.free(unsafe.Pointer(inputLayerName))

	outputLayerName := C.CString(p.options.OutputNode())
	defer C.free(unsafe.Pointer(outputLayerName))

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
	return nil
}

func (p *Predictor) EndProfiling() error {
	return nil
}

func (p *Predictor) DisableProfiling() error {
	return nil
}

func (p *Predictor) ReadProfile() (string, error) {
	return "", nil
}

func (p Predictor) Close() {
	C.DeleteTensorRT(p.ctx)
}
