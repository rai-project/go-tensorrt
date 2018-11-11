// +build linux,!ppc64le

package tensorrt

// #include <stdlib.h>
// #include "cbits/predictor.hpp"
import "C"
import (
	"context"
	"fmt"
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

func prod(arry []int) int {
	accum := int(1)
	for _, e := range arry {
		accum *= int(e)
	}
	return accum
}

func New(ctx context.Context, opts ...options.Option) (*Predictor, error) {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "new")
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

	inputNode := options.InputNodes()[0] // take the first input node
	if inputNode.Key() == "" {
		return nil, errors.New("expecting a valid (non-empty) input layer name")
	}
	inputNodeString := C.CString(inputNode.Key())
	defer C.free(unsafe.Pointer(inputNodeString))

	outputNode := options.OutputNode()
	if outputNode == "" {
		return nil, errors.New("expecting a valid (non-empty) output layer name")
	}
	outputNodeString := C.CString(outputNode)
	defer C.free(unsafe.Pointer(outputNodeString))

	return &Predictor{
		ctx: C.NewTensorRT(
			modelFileString,
			weightsFileString,
			C.int(options.BatchSize()),
			inputNodeString,
			outputNodeString,
			C.int(prod(inputNode.Shape())),
		),
		options: options,
	}, nil
}

func (p *Predictor) Predict(ctx context.Context, data []float32) error {
	if data == nil || len(data) < 1 {
		return fmt.Errorf("intput data nil or empty")
	}

	batchSize := p.options.BatchSize()
	shapeLen := int(C.GetShapeLenTensorRT(p.ctx))
	dataLen := len(data)

	inputCount := dataLen / shapeLen
	if batchSize > inputCount {
		padding := make([]float32, (batchSize-inputCount)*shapeLen)
		data = append(data, padding...)
	}

	ptr := (*C.float)(unsafe.Pointer(&data[0]))

	predictSpan, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "predict")
	C.PredictTensorRT(p.ctx, ptr)
	predictSpan.Finish()

	return nil
}

func (p *Predictor) ReadPredictionOutput(ctx context.Context) ([]float32, error) {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "read_prediction_output")
	defer span.Finish()

	batchSize := p.options.BatchSize()
	predLen := int(C.GetPredLenTensorRT(p.ctx))
	length := batchSize * predLen

	cPredictions := C.GetPredictionsTensorRT(p.ctx)
	if cPredictions == nil {
		return nil, errors.New("empty predictions")
	}

	slice := (*[1 << 30]C.float)(unsafe.Pointer(cPredictions))[:length:length]

	return slice, nil
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
