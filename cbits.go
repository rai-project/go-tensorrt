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

type Prediction struct {
	Index       string  `json:"index"`
	Probability float32 `json:"probability"`
}

type Predictions []Prediction

func (p *Predictor) Predict(imgArray []float32, width int, height int) (Predictions, error) {
	// check input

	ptr := (*C.float)(unsafe.Pointer(&imgArray[0]))
	r := C.PredictTensorRT(p.ctx, ptr, C.int(width), C.int(height))
	defer C.free(unsafe.Pointer(r))
	js := C.GoString(r)

	predictions := []Prediction{}
	err := json.Unmarshal([]byte(js), &predictions)
	if err != nil {
		return nil, err
	}
	return predictions, nil
}

func New(opts ...options.Option) (*Predictor, error) {
	options := options.New(opts...)
	modelFile := string(options.Graph())
	if !com.IsFile(modelFile) {
		return nil, errors.Errorf("file %s not found", modelFile)
	}
	weightsFile := string(options.Weights())
	if !com.IsFile(weightsFile) {
		return nil, errors.Errorf("file %s not found", weightsFile)
	}

	symbolFile := string(options.Class())
	if !com.IsFile(weightsFile) {
		return nil, errors.Errorf("file %s not found", weightsFile)
	}

	ctx := C.NewTensorRT(C.CString(modelFile), C.CString(weightsFile), C.int(options.BatchSize()), C.CString(symbolFile))
	return &Predictor{
		ctx:     ctx,
		options: options,
	}, nil
}
func (p Predictor) Close() {
	C.DeleteTensorRT(p.ctx)
}
