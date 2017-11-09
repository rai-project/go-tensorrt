package tensorrt

import (
	"math"
	"os"
	"testing"

	"github.com/rai-project/dlframework/framework/options"
	"github.com/rai-project/image"
)

func TestTensorRT(t *testing.T) {
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

	ctx, err := New(options.Class([]byte("networks/ilsvrc12_synset_words.txt")), options.Graph([]byte("networks/googlenet.prototxt")), options.Weights([]byte("networks/bvlc_googlenet.caffemodel")))
	if err != nil {
		t.Errorf("tensorRT initiate failed %v", err)
	}

	result, err := ctx.Predict(imgArray, w, h)
	if err != nil {
		t.Errorf("tensorRT inference failed %v", err)
	}
	if result[0].Index != "orange" {
		t.Errorf("tensorRT class label wrong")
	}
	if math.Abs(float64(result[0].Probability-0.972248)) > .001 {
		t.Errorf("tensorRT class probablity wrong")
	}
	ctx.Close()

}
