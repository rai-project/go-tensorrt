// +build linux
// +build !ppc64le
// +build !nogpu
// +build cgo

package tensorrt

// #include <stdlib.h>
// #include "cbits/predictor.hpp"
import "C"
import (
	"context"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/Unknwon/com"
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework/framework/options"
	"github.com/rai-project/tracer"
)

type Predictor struct {
	handle  C.PredictorHandle
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

	if len(options.InputNodes()) == 0 {
		return nil, errors.Errorf("input nodes not found")
	}
	if len(options.OutputNodes()) == 0 {
		return nil, errors.Errorf("output nodes not found")
	}

	modelFileString := C.CString(modelFile)
	defer C.free(unsafe.Pointer(modelFileString))

	weightsFileString := C.CString(weightsFile)
	defer C.free(unsafe.Pointer(weightsFileString))

	inputNodes := options.InputNodes() // take the first input node
	for _, n := range inputNodes {
		if n.Key == "" {
			return nil, errors.New("expecting a valid (non-empty) output layer name")
		}
	}

	cInputNodes := makeCStringArray(inputNodes)
	defer deleteCStringArray(cInputNodes)

	outputNodes := options.OutputNodes()
	for _, n := range outputNodes {
		if n.Key == "" {
			return nil, errors.New("expecting a valid (non-empty) output layer name")
		}
	}

	cOutputNodes := makeCStringArray(outputNodes)
	defer deleteCStringArray(cOutputNodes)

	handle := C.NewTensorRT(
		modelFileString,
		weightsFileString,
		C.int(options.BatchSize()),
		(**C.char)(unsafe.Pointer(&cInputNodes[0])),
		C.int(len(inputNodes)),
		(**C.char)(unsafe.Pointer(&cOutputNodes[0])),
		C.int(len(outputNodes)),
		C.int(prod(inputNodes[0].Shape)),
	)

	pred := &Predictor{
		handle:  handle,
		options: options,
	}

	runtime.SetFinalizer(pred, func(p *Predictor) {
		p.Close()
	})

	return pred, nil
}

func makeCStringArray(nds []options.Node) []*C.char {
	res := make([]*C.char, len(nds))
	for ii, nd := range nds {
		res[ii] = C.CString(nd.Key)
	}
	return res
}

func deleteCStringArray(strs []*C.char) {
	for ii := range strs {
		C.free(unsafe.Pointer(strs[ii]))
	}
}

func (p *Predictor) Predict(ctx context.Context, data []float32) error {
	if data == nil || len(data) < 1 {
		return fmt.Errorf("intput data nil or empty")
	}

	batchSize := p.options.BatchSize()
	shapeLen := int(C.GetShapeLenTensorRT(p.handle))
	dataLen := len(data)

	inputCount := dataLen / shapeLen
	if batchSize > inputCount {
		padding := make([]float32, (batchSize-inputCount)*shapeLen)
		data = append(data, padding...)
	}

	ptr := (*C.float)(unsafe.Pointer(&data[0]))

	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_predict")
	C.PredictTensorRT(p.handle, ptr)
	span.Finish()

	return nil
}

func (p *Predictor) ReadPredictionOutput(ctx context.Context) ([]float32, error) {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_read_prediction_output")
	defer span.Finish()

	batchSize := p.options.BatchSize()
	predLen := int(C.GetPredLenTensorRT(p.handle))
	length := batchSize * predLen

	cPredictions := C.GetPredictionsTensorRT(p.handle)
	if cPredictions == nil {
		return nil, errors.New("empty predictions")
	}

	slice := (*[1 << 30]float32)(unsafe.Pointer(cPredictions))[:length:length]

	return slice, nil
}

func (p *Predictor) Close() {
	var nilPredictorHandle C.PredictorHandle
	if p == nil || p.handle == nilPredictorHandle {
		return
	}
	C.DeleteTensorRT(p.handle)
	p.handle = nilPredictorHandle
}

func (p *Predictor) StartProfiling(name, metadata string) error {
	cname := C.CString(name)
	cmetadata := C.CString(metadata)
	defer C.free(unsafe.Pointer(cname))
	defer C.free(unsafe.Pointer(cmetadata))
	C.StartProfilingTensorRT(p.handle, cname, cmetadata)
	return nil
}

func (p *Predictor) EndProfiling() error {
	C.EndProfilingTensorRT(p.handle)
	return nil
}

func (p *Predictor) DisableProfiling() error {
	C.DisableProfilingTensorRT(p.handle)
	return nil
}

func (p *Predictor) ReadProfile() (string, error) {
	cstr := C.ReadProfileTensorRT(p.handle)
	if cstr == nil {
		return "", errors.New("failed to read nil profile")
	}
	defer C.free(unsafe.Pointer(cstr))
	return C.GoString(cstr), nil
}

func prod(arry []int) int {
	accum := int(1)
	for _, e := range arry {
		accum *= int(e)
	}
	return accum
}
