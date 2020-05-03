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

	"github.com/k0kubun/pp"

	"github.com/Unknwon/com"
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework/framework/options"
	"github.com/rai-project/tracer"
)

type Predictor struct {
	handle      C.PredictorHandle
	inputNodes  []options.Node
	outputNodes []options.Node
	options     *options.Options
}

func New(ctx context.Context, opts ...options.Option) (*Predictor, error) {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_new")
	defer span.Finish()

	options := options.New(opts...)

	modelFile := string(options.Graph())
	if !com.IsFile(modelFile) {
		return nil, errors.Errorf("file %s not found", modelFile)
	}

	if len(options.InputNodes()) == 0 {
		return nil, errors.Errorf("input nodes not found")
	}
	if len(options.OutputNodes()) == 0 {
		return nil, errors.Errorf("output nodes not found")
	}

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

	modelFileString := C.CString(modelFile)
	defer C.free(unsafe.Pointer(modelFileString))
	var weightsFileString *C.char
	format := ClassifyModelFormat(modelFile)
	if format == ModelFormatCaffe {
		weightsFile := string(options.Weights())
		if !com.IsFile(weightsFile) {
			return nil, errors.Errorf("file %s not found", weightsFile)
		}
		weightsFileString = C.CString(weightsFile)
		defer C.free(unsafe.Pointer(weightsFileString))
	}

	handle := C.NewTensorRTCaffePredictor(
		modelFileString,
		weightsFileString,
		C.TensorRT_DType(Float),
		(**C.char)(&cInputNodes[0]),
		C.int32_t(len(inputNodes)),
		(**C.char)(&cOutputNodes[0]),
		C.int32_t(len(outputNodes)),
		C.int32_t(options.BatchSize()),
	)

	pred := &Predictor{
		handle:      handle,
		inputNodes:  inputNodes,
		outputNodes: outputNodes,
		options:     options,
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

	cnamei := C.CString(p.inputNodes[0].Key)
	defer C.free(unsafe.Pointer(cnamei))

	cnameo := C.CString(p.outputNodes[0].Key)
	defer C.free(unsafe.Pointer(cnameo))

	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_predict")
	defer span.Finish()

	C.TensorRTPredictor_AddInput(
		p.handle,
		cnamei,
		C.TensorRT_DType(Float),
		unsafe.Pointer(&data[0]),
		C.size_t(len(data)),
	)

	C.TensorRTPredictor_AddOutput(
		p.handle,
		cnameo,
		C.TensorRT_DType(Float),
	)

	C.TensorRTPredictor_Run(p.handle)
	C.TensorRTPredictor_Synchronize(p.handle)
	return nil
}

func (p *Predictor) ReadPredictionOutputs(ctx context.Context) ([][]float32, error) {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_read_prediction_output")
	defer span.Finish()

	numOutputs := int(C.TensorRTPredictor_GetNumOutputs(p.handle))

	outputs := make([][]float32, numOutputs)
	for ii := 0; ii < numOutputs; ii++ {
		outputs[ii] = p.ReadPredictionOutput(p.outputNodes[ii].Key)
	}

	return outputs, nil
}

func prod(sz []int) int {
	res := 1
	for _, a := range sz {
		res *= a
	}
	return res
}

// ReadPredictionOutput ...
func (p *Predictor) ReadPredictionOutput(name string) []float32 {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	var ndims int32
	cdims := new(C.int32_t)

	data := C.TensorRTPredictor_GetOutput(p.handle, cname, (*C.int32_t)(&ndims), (**C.int32_t)(&cdims))

	dims := (*[1 << 30]C.int32_t)(unsafe.Pointer(cdims))[:ndims:ndims]

	sz := 1
	for ii := 0; ii < int(ndims); ii++ {
		sz *= int(dims[ii])
	}

	return (*[1 << 30]float32)(unsafe.Pointer(data))[:sz:sz]
}

func (p *Predictor) Close() {
	var nilPredictorHandle C.PredictorHandle
	if p == nil || p.handle == nilPredictorHandle {
		return
	}
	C.TensorRTPredictor_Delete(p.handle)
	p.handle = nilPredictorHandle
}

func (p *Predictor) StartProfiling(name, metadata string) error {
	cname := C.CString(name)
	cmetadata := C.CString(metadata)
	defer C.free(unsafe.Pointer(cname))
	defer C.free(unsafe.Pointer(cmetadata))
	C.TensorRTPredictor_StartProfiling(p.handle, cname, cmetadata)
	return nil
}

func (p *Predictor) EndProfiling() error {
	C.TensorRTPredictor_EndProfiling(p.handle)
	return nil
}

func (p *Predictor) ReadProfile() (string, error) {
	cstr := C.TensorRTPredictor_ReadProfiling(p.handle)
	if cstr == nil {
		return "", errors.New("failed to read nil profile")
	}
	defer C.free(unsafe.Pointer(cstr))
	return C.GoString(cstr), nil
}

func dummyPP() {
	pp.Println("dummy")
}
