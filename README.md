# go-tensorrt

[![Build Status](https://dev.azure.com/dakkak/rai/_apis/build/status/go-tensorrt)](https://dev.azure.com/dakkak/rai/_build/latest?definitionId=18)
[![Build Status](https://travis-ci.org/rai-project/go-tensorrt.svg?branch=master)](https://travis-ci.org/rai-project/go-tensorrt)[![Go Report Card](https://goreportcard.com/badge/github.com/rai-project/go-tensorrt)](https://goreportcard.com/report/github.com/rai-project/go-tensorrt)[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![](https://images.microbadger.com/badges/version/carml/go-tensorrt:amd64-cpu-latest.svg)](https://microbadger.com/images/carml/go-tensorrt:amd64-cpu-latest 'Get your own version badge on microbadger.com')
[![](https://images.microbadger.com/badges/version/carml/go-tensorrt:amd64-gpu-latest.svg)](https://microbadger.com/images/carml/go-tensorrt:amd64-gpu-latest 'Get your own version badge on microbadger.com')

Go binding for TensorRT C predict API.
This is used by the [TensorRT agent](https://github.com/rai-project/tensorrt) in [MLModelScope](mlmodelscope.org) to perform model inference in Go.

## Installation

Download and install go-tensorrt:

```
go get -v github.com/rai-project/go-tensorrt
```

The binding requires TensorRT and other Go packages.

### TensorRT

TensorRT currently only works in linux and requires GPU.
Please refer to [Installing TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html) to install TensorRT on your system.

**_Note_**: TensorRT is expected to be installed in either the system path or `/opt/tensorrt`.
See [lib.go](lib.go) for details.

If you get an error about not being able to write to `/opt` then perform the following

```
sudo mkdir -p /opt/tensorrt
sudo chown -R `whoami` /opt/tensorrt
```

If you are using TensorRT docker images or other libary paths, change CGO_CFLAGS, CGO_CXXFLAGS and CGO_LDFLAGS enviroment variables. Refer to [Using cgo with the go command](https://golang.org/cmd/cgo/#hdr-Using_cgo_with_the_go_command).

For example,

```
    export CGO_CFLAGS="${CGO_CFLAGS} -I/tmp/tensorrt/include"
    export CGO_CXXFLAGS="${CGO_CXXFLAGS} -I/tmp/tensorrt/include"
    export CGO_LDFLAGS="${CGO_LDFLAGS} -L/tmp/tensorrt/lib"
```


### Go Packages

You can install the dependency through `go get`.

```
cd $GOPATH/src/github.com/rai-project/tensorflow
go get -u -v ./...
```

Or use [Dep](https://github.com/golang/dep).

```
dep ensure -v
```

This installs the dependency in `vendor/`.


### Configure Environmental Variables

Configure the linker environmental variables since the TensorRT C library is under a non-system directory. Place the following in either your `~/.bashrc` or `~/.zshrc` file

Linux
```
export LIBRARY_PATH=$LIBRARY_PATH:/tensorrt/tensorrt/lib
export LD_LIBRARY_PATH=/opt/tensorrt/lib:$LD_LIBRARY_PATH
```

macOS
```
export LIBRARY_PATH=$LIBRARY_PATH:/opt/tensorrt/lib
export LD_LIBRARY_PATH=/opt/tensorrt/lib:$DYLD_LIBRARY_PATH
```

## Check the Build

Run `go build` in to check the dependences installation and library paths set-up.

**_Note_** : The CGO interface passes go pointers to the C API. This is an error by the CGO runtime. Disable the error by placing

```
export GODEBUG=cgocheck=0
```

in your `~/.bashrc` or `~/.zshrc` file and then run either `source ~/.bashrc` or `source ~/.zshrc`


## [Examples](examples)

The example shows how to use the MLModelScope tracer to profile the inference.
Refer to [Set up the external services](https://docs.mlmodelscope.org/installation/source/external_services/) to start the tracer.

If running on GPU, you can use nvprof to verify the profiling result.
Refer to [Profiler User's Guide](https://docs.nvidia.com/cuda/profiler-users-guide/index.html) for using nvprof.

