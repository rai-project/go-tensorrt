# Go Bindings for TensorRT
[![Build Status](https://dev.azure.com/dakkak/rai/_apis/build/status/go-tensorrt)](https://dev.azure.com/dakkak/rai/_build/latest?definitionId=18)
[![Build Status](https://travis-ci.org/rai-project/go-tensorrt.svg?branch=master)](https://travis-ci.org/rai-project/go-tensorrt)[![Go Report Card](https://goreportcard.com/badge/github.com/rai-project/go-tensorrt)](https://goreportcard.com/report/github.com/rai-project/go-tensorrt)[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![](https://images.microbadger.com/badges/version/carml/go-tensorrt:amd64-gpu-latest.svg)](https://microbadger.com/images/carml/go-tensorrt:amd64-gpu-latest 'Get your own version badge on microbadger.com')
[![](https://images.microbadger.com/badges/version/carml/go-tensorrt:ppc64le-gpu-latest.svg)](https://microbadger.com/images/carml/go-tensorrt:ppc64le-gpu-latest> 'Get your own version badge on microbadger.com')

## TensorRT Installation

**_Note_** TensorRT currently only works in linux and requires GPU.

Please refer to [Installing TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html) to install TensorRT on your system.
If you get an error about not being able to write to `/opt` then perform the following

```
sudo mkdir -p /opt/tensorrt
sudo chown -R `whoami` /opt/tensorrt
```

See [lib.go](lib.go) for details.

After installing TensorRT, run `export DYLD_LIBRARY_PATH=/opt/tensorrt/lib:$DYLD_LIBRARY_PATH` on mac os or `export LD_LIBRARY_PATH=/opt/tensorrt/lib:$DYLD_LIBRARY_PATH`on linux.

## Use Other Libary Paths

To use different library paths, change CGO_CFLAGS, CGO_CXXFLAGS and CGO_LDFLAGS enviroment variables.

For example,

```
    export CGO_CFLAGS="${CGO_CFLAGS} -I /usr/local/cuda-9.2/include -I/usr/local/cuda-9.2/nvvm/include -I /usr/local/cuda-9.2/extras/CUPTI/include -I /usr/local/cuda-9.2/targets/x86_64-linux/include -I /usr/local/cuda-9.2/targets/x86_64-linux/include/crt"
    export CGO_CXXFLAGS="${CGO_CXXFLAGS} -I /usr/local/cuda-9.2/include -I/usr/local/cuda-9.2/nvvm/include -I /usr/local/cuda-9.2/extras/CUPTI/include -I /usr/local/cuda-9.2/targets/x86_64-linux/include -I /usr/local/cuda-9.2/targets/x86_64-linux/include/crt"
    export CGO_LDFLAGS="${CGO_LDFLAGS} -L /usr/local/nvidia/lib64 -L /usr/local/cuda-9.2/nvvm/lib64 -L /usr/local/cuda-9.2/lib64 -L /usr/local/cuda-9.2/lib64/stubs -L /usr/local/cuda-9.2/targets/x86_64-linux/lib/stubs/ -L /usr/local/cuda-9.2/lib64/stubs -L /usr/local/cuda-9.2/extras/CUPTI/lib64"
```

Run `go build` in to check the tensorrt installation and library paths set-up.

## Run The Examples

Make sure you have already [install mlmodelscope dependences](https://docs.mlmodelscope.org/installation/source/dependencies/) and [set up the external services](https://docs.mlmodelscope.org/installation/source/external_services/).

### batch

This example is to show how to use mlmodelscope tracing to profile the inference.

```
  cd example/batch
  go build
  ./batch
```

Then you can go to `localhost:16686` to look at the trace of that inference.

### batch_nvprof

This example is to show how to use nvprof to profile the inference.

```
  cd example/batch_nvprof
  go build
  nvprof --profile-from-start off ./batch_nvprof
```

Refer to [Profiler User's Guide](https://docs.nvidia.com/cuda/profiler-users-guide/index.html) on how to use nvprof.
