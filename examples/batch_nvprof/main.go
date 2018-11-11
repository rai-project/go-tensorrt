package main

// #cgo linux CFLAGS: -I/usr/local/cuda/include
// #cgo linux LDFLAGS: -lcuda -lcudart -L/usr/local/cuda/lib64
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_profiler_api.h>
import "C"

import (
	"bufio"
	"context"
	"fmt"
	"image"
	"os"
	"path/filepath"

	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/k0kubun/pp"

	"github.com/rai-project/config"
	"github.com/rai-project/dlframework/framework/options"
	"github.com/rai-project/downloadmanager"
	"github.com/rai-project/go-tensorrt"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tracer"
	_ "github.com/rai-project/tracer/all"

	_ "github.com/rai-project/tracer/jaeger"
)

var (
	batchSize    = 64
	graph_url    = "https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt"
	weights_url  = "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel"
	features_url = "http://data.dmlc.ml/mxnet/models/imagenet/synset.txt"
)

// convert go Image to 1-dim array
func cvtImageTo1DArray(src image.Image, mean []float32) ([]float32, error) {
	if src == nil {
		return nil, fmt.Errorf("src image nil")
	}

	b := src.Bounds()
	h := b.Max.Y - b.Min.Y // image height
	w := b.Max.X - b.Min.X // image width

	res := make([]float32, 3*h*w)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := src.At(x+b.Min.X, y+b.Min.Y).RGBA()
			res[y*w+x] = float32(b>>8) - mean[0]
			res[w*h+y*w+x] = float32(g>>8) - mean[1]
			res[2*w*h+y*w+x] = float32(r>>8) - mean[2]
		}
	}

	return res, nil
}

func main() {
	defer tracer.Close()

	dir, _ := filepath.Abs("../tmp")
	graph := filepath.Join(dir, "deploy.prototxt")
	weights := filepath.Join(dir, "bvlc_alexnet.caffemodel")
	features := filepath.Join(dir, "synset.txt")

	if _, err := os.Stat(graph); os.IsNotExist(err) {
		if _, err := downloadmanager.DownloadInto(graph_url, dir); err != nil {
			panic(err)
		}
	}
	if _, err := os.Stat(weights); os.IsNotExist(err) {

		if _, err := downloadmanager.DownloadInto(weights_url, dir); err != nil {
			panic(err)
		}
	}
	if _, err := os.Stat(features); os.IsNotExist(err) {

		if _, err := downloadmanager.DownloadInto(features_url, dir); err != nil {
			panic(err)
		}
	}

	imgDir, _ := filepath.Abs("../_fixtures")
	imagePath := filepath.Join(imgDir, "platypus.jpg")

	img, err := imgio.Open(imagePath)
	if err != nil {
		panic(err)
	}

	var input []float32
	for ii := 0; ii < batchSize; ii++ {
		resized := transform.Resize(img, 227, 227, transform.Linear)
		res, err := cvtImageTo1DArray(resized, []float32{123, 117, 104})
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

	span, ctx := tracer.StartSpanFromContext(ctx, tracer.FULL_TRACE, "tensorrt_batch_nvporf")
	defer span.Finish()

	predictor, err := tensorrt.New(
		ctx,
		options.WithOptions(opts),
		options.Device(device, 0),
		options.Graph([]byte(graph)),
		options.Weights([]byte(weights)),
		options.InputNode("data", []int{3, 227, 227}),
		options.OutputNode("prob"),
		options.BatchSize(batchSize),
	)
	if err != nil {
		panic(err)
	}
	defer predictor.Close()

	err = predictor.Predict(ctx, input)
	if err != nil {
		panic(err)
	}

	C.cudaProfilerStart()

	err = predictor.Predict(ctx, input)
	if err != nil {
		panic(err)
	}

	C.cudaDeviceSynchronize()
	C.cudaProfilerStop()

	predictions, err := predictor.ReadPredictions(ctx)
	if err != nil {
		panic(err)
	}

	if true {
		var labels []string
		f, err := os.Open(features)
		if err != nil {
			panic(err)
		}
		defer f.Close()
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			line := scanner.Text()
			labels = append(labels, line)
		}

		len := len(predictions) / batchSize
		for i := 0; i < 1; i++ {
			res := predictions[i*len : (i+1)*len]
			res.Sort()
			pp.Println(res[0].Probability)
			pp.Println(labels[res[0].Index])
		}
	} else {
		_ = predictions
	}
}

func init() {
	config.Init(
		config.AppName("carml"),
		config.VerboseMode(true),
		config.DebugMode(true),
	)
}
