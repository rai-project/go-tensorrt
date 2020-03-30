// +build linux
// +build !ppc64le
// +build !nogpu
// +build cgo

package tensorrt

import (
	"context"
	"fmt"
	"image"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/GeertJohan/go-sourcepath"
	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/k0kubun/pp"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/feature"
	"github.com/rai-project/dlframework/framework/options"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	_ "github.com/rai-project/tracer/all"
	"github.com/stretchr/testify/assert"
	gotensor "gorgonia.org/tensor"
)

var (
	batchSize            = 1
	shape                = []int{1, 3, 224, 224}
	mean                 = []float32{123.68, 116.779, 103.939}
	scale                = []float32{1.0, 1.0, 1.0}
	thisDir              = sourcepath.MustAbsoluteDir()
	imgPath              = filepath.Join(thisDir, "_fixtures", "platypus.jpg")
	labelFilePath        = filepath.Join(thisDir, "_fixtures", "resnet50", "synset.txt")
	caffeGraphFilePath   = filepath.Join(thisDir, "_fixtures", "resnet50", "resnet50.prototxt")
	caffeWeightsFilePath = filepath.Join(thisDir, "_fixtures", "resnet50", "resnet50.caffemodel")
	onnxModelPath        = filepath.Join(thisDir, "_fixtures", "ResNet50.onnx")
	uffModelPath         = filepath.Join(thisDir, "_fixtures", "resnet50-infer-5.uff")
)

// convert go RGB Image to 1D normalized RGB array
func cvtRGBImageToNCHW1DArray(src image.Image, mean []float32, scale []float32) ([]float32, error) {
	if src == nil {
		return nil, fmt.Errorf("src image nil")
	}

	in := src.Bounds()
	height := in.Max.Y - in.Min.Y // image height
	width := in.Max.X - in.Min.X  // image width
	stride := width * height      // image size per channel

	out := make([]float32, 3*height*width)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := src.At(x+in.Min.X, y+in.Min.Y).RGBA()
			out[0*stride+y*width+x] = (float32(r>>8) - mean[0]) / scale[0]
			out[1*stride+y*width+x] = (float32(g>>8) - mean[1]) / scale[1]
			out[2*stride+y*width+x] = (float32(b>>8) - mean[2]) / scale[2]
		}
	}

	return out, nil
}

func TestTensorRTCaffe(t *testing.T) {
	img, err := imgio.Open(imgPath)
	if err != nil {
		t.Errorf("Test input image is not found: %v", err)
	}

	height := shape[2]
	width := shape[3]

	var input []float32
	for ii := 0; ii < batchSize; ii++ {
		resized := transform.Resize(img, height, width, transform.Linear)
		res, err := cvtRGBImageToNCHW1DArray(resized, mean, scale)
		if err != nil {
			t.Errorf("Test input image transformation is not successful: %v", err)
		}
		input = append(input, res...)
	}

	opts := options.New()

	if !nvidiasmi.HasGPU {
		t.Errorf("GPU is not detected: %v", err)
	}
	device := options.CUDA_DEVICE

	ctx := context.Background()
	in := options.Node{
		Key:   "data",
		Shape: shape,
		Dtype: gotensor.Float32,
	}
	out := options.Node{
		Key:   "prob",
		Dtype: gotensor.Float32,
	}

	predictor, err := New(
		ctx,
		options.WithOptions(opts),
		options.Device(device, 0),
		options.Graph([]byte(caffeGraphFilePath)),
		options.Weights([]byte(caffeWeightsFilePath)),
		options.BatchSize(batchSize),
		options.InputNodes([]options.Node{in}),
		options.OutputNodes([]options.Node{out}),
	)
	fmt.Println("here here here here here")
	if err != nil {
		t.Errorf("TensorRT predictor initiation failed %v", err)
	}

	defer predictor.Close()

	err = predictor.Predict(ctx, input)
	if err != nil {
		t.Errorf("tensorRT inference failed %v", err)
	}

	outputs, err := predictor.ReadPredictionOutputs(ctx)
	if err != nil {
		panic(err)
	}

	output := outputs[0]
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

	assert.Equal(t, int32(103), top1.GetClassification().GetIndex())
	pp.Println(top1.GetClassification().GetLabel(), top1.GetProbability())
	if top1.GetClassification().GetLabel() != "n01873310 platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus" {
		t.Errorf("tensorRT class label wrong")
	}
	if math.Abs(float64(top1.GetProbability()-0.99)) > .01 {
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
