package main

// #include "cbits/test.hpp"
import "C"
import (
	"fmt"
)

func main() {
	y := int(C.Start_code(C.int(5)))
	fmt.Printf("Hello, word. %d\n", y)
}

// type Device int

// const (
// 	CPUDevice  Device = Device(C.CPU_DEVICE_KIND)
// 	CUDADevice        = Device(C.CUDA_DEVICE_KIND)
// )

// type Predictor struct {
// 	device C.DeviceKind
// 	ctx    C.PredictorContext
// }

// func init() {
// 	config.AfterInit(func() {
// 		var device C.DeviceKind = C.CPU_DEVICE_KIND
// 		nvidiasmi.Wait()
// 		if nvidiasmi.HasGPU {
// 			device = C.CUDA_DEVICE_KIND
// 		}
// 		C.InitCaffe2(device)
// 	})
// }

// func New(opts ...options.Option) (*Predictor, error) {
// 	options := options.New(opts...)
// 	initNetFile := string(options.Graph())
// 	if !com.IsFile(initNetFile) {
// 		return nil, errors.Errorf("file %s not found", initNetFile)
// 	}
// 	predictNetFile := string(options.Weights())
// 	if !com.IsFile(predictNetFile) {
// 		return nil, errors.Errorf("file %s not found", predictNetFile)
// 	}
// 	device := C.DeviceKind(CPUDevice)
// 	if options.UsesGPU() {
// 		if !nvidiasmi.HasGPU {
// 			return nil, errors.New("no GPU device")
// 		}
// 		device = C.DeviceKind(CUDADevice)
// 	}
// 	ctx := C.NewCaffe2(C.CString(initNetFile), C.CString(predictNetFile), device)
// 	if ctx == nil {
// 		return nil, errors.New("unable to create caffe2 predictor context")
// 	}
// 	return &Predictor{
// 		device: device,
// 		ctx:    ctx,
// 	}, nil
// }

// func (p *Predictor) StartProfiling(name, metadata string) error {
// 	cname := C.CString(name)
// 	cmetadata := C.CString(metadata)
// 	defer C.free(unsafe.Pointer(cname))
// 	defer C.free(unsafe.Pointer(cmetadata))
// 	C.StartProfilingCaffe2(p.ctx, cname, cmetadata, p.device)
// 	return nil
// }

// func (p *Predictor) EndProfiling() error {
// 	C.EndProfilingCaffe2(p.ctx, p.device)
// 	return nil
// }

// func (p *Predictor) DisableProfiling() error {
// 	C.DisableProfilingCaffe2(p.ctx, p.device)
// 	return nil
// }

// func (p *Predictor) ReadProfile() (string, error) {
// 	cstr := C.ReadProfileCaffe2(p.ctx, p.device)
// 	if cstr == nil {
// 		return "", errors.New("failed to read nil profile")
// 	}
// 	defer C.free(unsafe.Pointer(cstr))
// 	return C.GoString(cstr), nil
// }

// func (p *Predictor) Predict(data []float32, batchSize int, channels int,
// 	width int, height int) (Predictions, error) {
// 	// check input
// 	if data == nil || len(data) < 1 {
// 		return nil, fmt.Errorf("intput data nil or empty")
// 	}

// 	if batchSize != 1 {
// 		dataLen := int64(len(data))
// 		shapeLen := int64(width * height * channels)
// 		inputCount := dataLen / shapeLen
// 		padding := make([]float32, (int64(batchSize)-inputCount)*shapeLen)
// 		data = append(data, padding...)
// 	}

// 	ptr := (*C.float)(unsafe.Pointer(&data[0]))
// 	r := C.PredictCaffe2(p.ctx, ptr, C.int(batchSize), C.int(channels), C.int(width), C.int(height), p.device)
// 	defer C.free(unsafe.Pointer(r))
// 	js := C.GoString(r)

// 	predictions := []Prediction{}
// 	err := json.Unmarshal([]byte(js), &predictions)
// 	if err != nil {
// 		return nil, err
// 	}
// 	return predictions, nil
// }

// func (p *Predictor) Close() {
// 	C.DeleteCaffe2(p.ctx, p.device)
// }
