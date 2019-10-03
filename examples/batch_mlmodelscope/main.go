package main

import (
	"bufio"
	"context"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"sort"

	"github.com/Unknwon/com"
	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/k0kubun/pp"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/feature"
	"github.com/rai-project/dlframework/framework/options"
	"github.com/rai-project/downloadmanager"
	cupti "github.com/rai-project/go-cupti"
	"github.com/rai-project/go-tensorrt"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tracer"
	_ "github.com/rai-project/tracer/all"
	"github.com/rai-project/tracer/ctimer"
	gotensor "gorgonia.org/tensor"
)

var (
	batchSize   = 1
	model       = "resnet50"
	shape       = []int{1, 3, 224, 224}
	mean        = []float32{123.68, 116.779, 103.939}
	scale       = []float32{1.0, 1.0, 1.0}
	imgDir, _   = filepath.Abs("../_fixtures")
	imgPath     = filepath.Join(imgDir, "platypus.jpg")
	graph_url   = "http://s3.amazonaws.com/store.carml.org/models/caffe/resnet50/ResNet-50-deploy.prototxt"
	weights_url = "http://s3.amazonaws.com/store.carml.org/models/caffe/resnet50/ResNet-50-model.caffemodel"
	synset_url  = "http://s3.amazonaws.com/store.carml.org/synsets/imagenet/synset1001.txt"
)

// convert go Image to 1-dim array
func cvtRGBImageToNCHW1DArray(src image.Image, mean []float32, scale []float32) ([]float32, error) {
	if src == nil {
		return nil, fmt.Errorf("src image nil")
	}

	in := src.Bounds()
	height := in.Max.Y - in.Min.Y // image height
	width := in.Max.X - in.Min.X  // image width

	out := make([]float32, 3*height*width)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := src.At(x+in.Min.X, y+in.Min.Y).RGBA()
			out[y*width+x] = (float32(b) - mean[2]) / scale[2]
			out[width*height+y*width+x] = (float32(g) - mean[1]) / scale[1]
			out[2*width*height+y*width+x] = (float32(r) - mean[0]) / scale[0]
		}
	}

	return out, nil
}

func main() {
  defer tracer.Close()
  
	baseDir, _ := filepath.Abs("../tmp")
	dir := filepath.Join(baseDir, model)
	graph := filepath.Join(dir, model+".prototxt")
	weights := filepath.Join(dir, model+".caffemodel")
	synset := filepath.Join(dir, "synset.txt")

	if !com.IsFile(graph) {
		if _, _, err := downloadmanager.DownloadFile(graph_url, graph); err != nil {
			panic(err)
		}
	}
	if !com.IsFile(weights) {
		if _, _, err := downloadmanager.DownloadFile(weights_url, weights); err != nil {
			panic(err)
		}
	}
	if !com.IsFile(synset) {
		if _, _, err := downloadmanager.DownloadFile(synset_url, synset); err != nil {
			panic(err)
		}
	}

	img, err := imgio.Open(imgPath)
	if err != nil {
		panic(err)
	}

	height := shape[2]
	width := shape[3]

	var input []float32
	for ii := 0; ii < batchSize; ii++ {
		resized := transform.Resize(img, height, width, transform.Linear)
		res, err := cvtRGBImageToNCHW1DArray(resized, mean, scale)
		if err != nil {
			panic(err)
		}
		input = append(input, res...)
	}

	opts := options.New()

	if !nvidiasmi.HasGPU {
		panic("no GPU")
	}
	device := options.CUDA_DEVICE

	ctx := context.Background()

	span, ctx := tracer.StartSpanFromContext(ctx, tracer.FULL_TRACE, "tensorrt_batch")
	defer span.Finish()

	in := options.Node{
		Key:   "data",
		Shape: shape,
		Dtype: gotensor.Float32,
	}

	predictor, err := tensorrt.New(
		ctx,
		options.WithOptions(opts),
		options.Device(device, 0),
		options.Graph([]byte(graph)),
		options.Weights([]byte(weights)),
		options.BatchSize(batchSize),
		options.InputNodes([]options.Node{in}),
		options.OutputNodes([]options.Node{
			options.Node{
				Key:   "prob",
				Dtype: gotensor.Float32,
			},
		}),
	)
	if err != nil {
		panic(fmt.Sprintf("%v", err))
	}
	defer predictor.Close()

	for ii:=0; ii < 3; ii++ {
	err = predictor.Predict(ctx, input)
	if err != nil {
		panic(err)
	}
}

	enableCupti := true 
	var cu *cupti.CUPTI
	if enableCupti {
		cu, err = cupti.New(cupti.Context(ctx))
		if err != nil {
			panic(err)
		}
	}

	predictor.StartProfiling("predict", "")

	err = predictor.Predict(ctx, input)
	if err != nil {
		panic(err)
	}

	predictor.EndProfiling()

	if enableCupti {
		cu.Wait()
		cu.Close()
	}

	if true {
		profBuffer, err := predictor.ReadProfile()
		if err != nil {
			panic(err)
		}

		t, err := ctimer.New(profBuffer)
		if err != nil {
			panic(err)
		}
		t.Publish(ctx, tracer.FRAMEWORK_TRACE)

		outputs, err := predictor.ReadPredictionOutputs(ctx)
		if err != nil {
			panic(err)
		}

		output := outputs[0]

		var labels []string
		f, err := os.Open(synset)
		if err != nil {
			panic(err)
		}
		defer f.Close()
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			line := scanner.Text()
			labels = append(labels, line)
		}

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

		if true {
			results := features[0]
			for i := 0; i < 3; i++ {
				prediction := results[i]
				pp.Println(prediction.Probability)
				pp.Println(prediction.GetClassification().GetIndex())
				pp.Println(prediction.GetClassification().GetLabel())
			}
		} else {
			_ = features
		}
	}
}

func init() {
	config.Init(
		config.AppName("carml"),
		config.VerboseMode(true),
		config.DebugMode(true),
	)
}
