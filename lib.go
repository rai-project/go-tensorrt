// +build linux

package tensorrt

// #cgo LDFLAGS: -lnvinfer -lnvcaffe_parser -lcudart -L${SRCDIR} -lstdc++
// #cgo CXXFLAGS: -std=c++11  -O3 -Wall -g
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function
// #cgo linux,amd64 CXXFLAGS: -I/usr/local/cuda-9.0/include -I/usr/include/x86_64-linux-gnu -I/opt/frameworks/tensorrt/include
// #cgo linux,arm64 CXXFLAGS: -I/usr/local/cuda-9.0/include -I/usr/include/aarch64-linux-gnu -I/opt/frameworks/tensorrt/include
// #cgo CXXFLAGS: -I${SRCDIR}/cbits
// #cgo linux,amd64 LDFLAGS: -L/usr/local/cuda/lib64 -L/opt/frameworks/tensorrt/lib -lcudart
// #cgo linux,arm64 LDFLAGS: -L/usr/local/cuda-9.0/lib64 -L/opt/frameworks/tensorrt/lib -lcudart
import "C"
