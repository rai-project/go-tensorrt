// +build linux
// +build !ppc64le
// +build !nogpu
// +build cgo
package tensorrt

import (
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/GeertJohan/go-sourcepath"
	"github.com/k0kubun/pp"
	"github.com/stretchr/testify/assert"

	"github.com/rai-project/dlframework/framework/options"
	"github.com/rai-project/image"
	"github.com/rai-project/image/types"
)

var (
	thisDir         = sourcepath.MustAbsoluteDir()
	classFilePath   = filepath.Join(thisDir, "_fixtures", "networks", "ilsvrc12_synset_words.txt")
	graphFilePath   = filepath.Join(thisDir, "_fixtures", "networks", "googlenet.prototxt")
	weightsFilePath = filepath.Join(thisDir, "_fixtures", "networks", "bvlc_googlenet.caffemodel")
)

func TestTensorRT(t *testing.T) {

	reader, _ := os.Open(filepath.Join(thisDir, "_fixtures", "cat.jpg"))
	defer reader.Close()

	img0, err := image.Read(reader, image.Resized(224, 224))
	assert.NoError(t, err)

	img := img0.(*types.RGBImage)

	const channels = 3
	bounds := img.Bounds()
	w, h := bounds.Max.X, bounds.Max.Y
	imgArray := make([]float32, w*h*3)
	pixels := img.Pix

	mean := []float32{104.0069879317889, 116.66876761696767, 122.6789143406786}
	scale := float32(1.0)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			width, height := w, h
			offset := y*img.Stride + x*3
			rgb := pixels[offset : offset+3]
			r, g, b := rgb[0], rgb[1], rgb[2]
			imgArray[y*width+x] = (float32(b) - mean[0]) / scale
			imgArray[width*height+y*width+x] = (float32(g) - mean[1]) / scale
			imgArray[2*width*height+y*width+x] = (float32(r) - mean[2]) / scale

		}
	}

	pred, err := New(
		options.Graph([]byte(graphFilePath)),
		options.Weights([]byte(weightsFilePath)),
		options.BatchSize(1),
		options.InputNode("data", []uint32{3, 224, 224}),
		options.OutputNode("prob"),
	)
	if err != nil {
		t.Errorf("tensorRT initiate failed %v", err)
	}

	defer pred.Close()

	result, err := pred.Predict("data", "prob", imgArray, []uint32{3, 224, 224})
	if err != nil {
		t.Errorf("tensorRT inference failed %v", err)
	}

	result.Sort()

	classesFileContent, err := ioutil.ReadFile(classFilePath)
	assert.NoError(t, err)

	classes := strings.Split(string(classesFileContent), "\n")

	// pp.Println(result[:10])
	assert.Equal(t, 281, result[0].Index)

	pp.Println(result[0])
	if classes[result[0].Index] != "n02123045 tabby, tabby cat" {
		t.Errorf("tensorRT class label wrong")
	}
	if math.Abs(float64(result[0].Probability-0.5)) < .001 {
		t.Errorf("tensorRT class probablity wrong")
	}
}
