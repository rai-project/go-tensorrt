package main

// #include <stdlib.h>
// #include "cbits/predict.hpp"
import "C"
import (
	"encoding/json"
	"fmt"
	"os"
	"unsafe"

	"github.com/rai-project/image"
)

type Prediction struct {
	Index       string  `json:"index"`
	Probability float32 `json:"probability"`
}

type Predictions []Prediction

func main() {
	reader, _ := os.Open("Orange.jpg")
	defer reader.Close()

	img, _ := image.Read(reader)

	bounds := img.Bounds()
	w, h := bounds.Max.X, bounds.Max.Y
	imgArray := make([]float32, w*h*4)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			// for y := 0; y < h; y++ {
			imgColor := img.At(x, y)
			a, b, c, d := imgColor.RGBA()
			imgArray[y*w*4+x*4] = float32(a >> 8)
			imgArray[y*w*4+x*4+1] = float32(b >> 8)
			imgArray[y*w*4+x*4+2] = float32(c >> 8)
			imgArray[y*w*4+x*4+3] = float32(d >> 8)
		}
	}
	ctx := C.NewTensorRT()
	// ptr := (*C.float)(unsafe.Pointer(&imgArray[0]))
	// r := C.Start_code(ptr, C.int(w), C.int(h))
	result, _ := Predict(ctx, imgArray, w, h)
	// y := int(C.Start_code(C.int(5)))
	fmt.Printf("Hello, word.\n %v\n", result)
}
func Predict(ctx C.PredictorContext, imgArray []float32, width int, height int) (Predictions, error) {
	// check input

	ptr := (*C.float)(unsafe.Pointer(&imgArray[0]))
	r := C.PredictTensorRT(ctx, ptr, C.int(width), C.int(height))
	defer C.free(unsafe.Pointer(r))
	js := C.GoString(r)

	predictions := []Prediction{}
	err := json.Unmarshal([]byte(js), &predictions)
	if err != nil {
		return nil, err
	}
	return predictions, nil
}
