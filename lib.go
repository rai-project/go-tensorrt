// +build linux

package tensorrt

// #cgo LDFLAGS: -lstdc++ -lnvinfer -lnvcaffe_parser
// #cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/cbits -O3 -Wall -g
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function
// #cgo linux,!ppc64le CXXFLAGS: -I/usr/local/cuda/include
// #cgo linux,amd64 CXXFLAGS: -I/opt/tensorrt/include -I/usr/include/x86_64-linux-gnu
// #cgo linux,arm64 CXXFLAGS: -I/usr/include/aarch64-linux-gnu
// #cgo linux,!ppc64le LDFLAGS: -L/usr/local/cuda/lib64 -L/opt/tensorrt/lib -L/usr/lib/x86_64-linux-gnu -L/usr/lib/aarch64-linux-gnu -lcudnn -lcudart
import "C"
