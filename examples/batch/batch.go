package main

import (
	"bufio"
	"context"
	"fmt"
	"image"
	"os"
	"path/filepath"

	"github.com/GeertJohan/go-sourcepath"

	"github.com/k0kubun/pp"

	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework/framework/options"
	"github.com/rai-project/downloadmanager"
	"github.com/rai-project/go-tensorrt"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tracer"
	_ "github.com/rai-project/tracer/all"
	"github.com/rai-project/tracer/ctimer"
)

var (
	batch        = 64
	graph_url    = "https://github.com/DeepScale/SqueezeNet/raw/master/SqueezeNet_v1.0/deploy.prototxt"
	weights_url  = "https://github.com/DeepScale/SqueezeNet/raw/master/SqueezeNet_v1.0/squeezenet_v1.0.caffemodel"
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
	dir, _ := filepath.Abs("../tmp")
	graph := filepath.Join(dir, "deploy.prototxt")
	weights := filepath.Join(dir, "bvlc_alexnet.caffemodel")
	features := filepath.Join(dir, "synset.txt")

  defer tracer.Close()

  span, ctx := tracer.StartSpanFromContext(context.Background(), tracer.FULL_TRACE, "tensorrt_single")
	defer span.Finish()

	if _, err := downloadmanager.DownloadInto(graph_url, dir); err != nil {
		os.Exit(-1)
	}

	if _, err := downloadmanager.DownloadInto(weights_url, dir); err != nil {
		os.Exit(-1)
	}
	if _, err := downloadmanager.DownloadInto(features_url, dir); err != nil {
		os.Exit(-1)
	}


	var input []float32
	cnt := 0

	imgDir, _ := filepath.Abs("../_fixtures")
	err := filepath.Walk(imgDir, func(path string, info os.FileInfo, err error) error {
		if path == imgDir || filepath.Ext(path) != ".jpg" || cnt >= batch {
			return nil
		}

		img, err := imgio.Open(path)
		if err != nil {
			return err
		}
		resized := transform.Resize(img, 227, 227, transform.Linear)
		res, err := cvtImageTo1DArray(resized, {123, 117, 104})
		if err != nil {
			panic(err)
		}
		input = append(input, res...)
		cnt++

		return nil
  })

	if err != nil {
		panic(err)
	}

	opts := options.New()

	device := options.CPU_DEVICE
	if nvidiasmi.HasGPU {
		device = options.CUDA_DEVICE
	} else {
		panic("no gpu")
  }

	// pp.Println("Using device = ", device)

	// create predictor
	predictor, err := tensorrt.New(
		options.WithOptions(opts),
		options.Device(device, 0),
		options.Graph([]byte(graph)),
		options.Weights([]byte(weights)),
		options.BatchSize(uint32(batch)))
	if err != nil {
		panic(err)
  }
	defer predictor.Close()

  predictor.StartProfiling("predict", "")
	predictions, err := predictor.Predict(ctx, input)
	if err != nil {
		pp.Println(err)
		os.Exit(-1)
	}
	predictor.EndProfiling()

  profBuffer, err := predictor.ReadProfile()
	if err != nil {
		pp.Println(err)
		os.Exit(-1)
	}

	t, err := ctimer.New(profBuffer)
	if err != nil {
		pp.Println(err)
		os.Exit(-1)
	}
	t.Publish(ctx)
	predictor.DisableProfiling()

	var labels []string
	f, err := os.Open(features)
	if err != nil {
		os.Exit(-1)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		labels = append(labels, line)
	}

	len := len(predictions) / batch
	for i := 0; i < cnt; i++ {
		res := predictions[i*len : (i+1)*len]
		res.Sort()
		pp.Println(res[0].Probability)
		pp.Println(labels[res[0].Index])
	}

	// os.RemoveAll(dir)
}

func init() {
	config.Init(
		config.AppName("carml"),
		config.VerboseMode(true),
		config.DebugMode(true),
	)
}
