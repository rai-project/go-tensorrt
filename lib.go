package main

// #cgo LDFLAGS: -lnvinfer -lnvcaffe_parser -lcudart -L${SRCDIR} -lcudaUtil -L/usr/local/cuda/lib64 -lstdc++
// #cgo CXXFLAGS: -std=c++11  -O3 -Wall -g
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function
// #cgo CXXFLAGS: -I/usr/local/cuda/include -I/usr/include/x86_64-linux-gnu
// #cgo CXXFLAGS: -I${SRCDIR}/cbits -I${SRCDIR}/cbits/util -I${SRCDIR}/cuda
import "C"
