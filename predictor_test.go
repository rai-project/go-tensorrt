// +build linux
// +build !ppc64le
// +build !nogpu
// +build cgo

package tensorrt

import (
	"context"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/GeertJohan/go-sourcepath"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/feature"
	"github.com/rai-project/dlframework/framework/options"
	"github.com/rai-project/image"
	"github.com/rai-project/image/types"
	"github.com/stretchr/testify/assert"
)

var (
	batchSize       = 1
	thisDir         = sourcepath.MustAbsoluteDir()
	labelFilePath   = filepath.Join(thisDir, "_fixtures", "networks", "ilsvrc12_synset_words.txt")
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

	ctx := context.Background()
	predictor, err := New(
		ctx,
		options.Graph([]byte(graphFilePath)),
		options.Weights([]byte(weightsFilePath)),
		options.BatchSize(1),
		options.InputNode("data", []int{3, 224, 224}),
		options.OutputNode("prob"),
	)
	if err != nil {
		t.Errorf("tensorRT initiate failed %v", err)
	}

	defer predictor.Close()

	err = predictor.Predict(ctx, imgArray)
	if err != nil {
		t.Errorf("tensorRT inference failed %v", err)
	}

	output, err := predictor.ReadPredictionOutput(ctx)
	if err != nil {
		panic(err)
	}

	labelsFileContent, err := ioutil.ReadFile(labelFilePath)
	assert.NoError(t, err)
	labels := strings.Split(string(labelsFileContent), "\n")

	features := make([]dlframework.Features, batchSize)
	featuresLen := len(output) / batchSize

	for ii := 0; ii < batchSize; ii++ {
		rprobs := make([]*dlframework.Feature, featuresLen)
		for jj := 0; jj < featuresLen; jj++ {
			rprobs[jj] = feature.New(
				feature.ClassificationIndex(int32(jj)),
				feature.ClassificationLabel(labels[jj]),
				feature.Probability(output[ii*featuresLen+jj]),
			)
		}
		sort.Sort(dlframework.Features(rprobs))
		features[ii] = rprobs
	}

	top1 := features[0][0]

	assert.Equal(t, int32(281), top1.GetClassification().GetIndex())

	if top1.GetClassification().GetLabel() != "n02123045 tabby, tabby cat" {
		t.Errorf("tensorRT class label wrong")
	}
	if math.Abs(float64(top1.GetProbability()-0.324)) > .001 {
		t.Errorf("tensorRT class probablity wrong")
	}
}

func TestMain(m *testing.M) {
	config.Init(
		config.AppName("carml"),
		config.VerboseMode(true),
		config.DebugMode(true),
	)

	os.Exit(m.Run())
}
