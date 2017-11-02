package tensorrt

// #cgo LDFLAGS: -lnvinfer -lnvcaffe_parser -lQtCore -lQtGui -lcudart -L${SRCDIR}/cuda -lcudaUtil -L/usr/local/cuda/lib64
// #cgo CXXFLAGS: -std=c++11  -O3 -Wall -g
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function
// #cgo CXXFLAGS: -I/usr/include/qt4/QtGui -I/usr/include/qt4/QtCore -I/usr/include/qt4 -I/usr/local/cuda/include
// #cgo CXXFLAGS: -I${SRCDIR}/cbits -I${SRCDIR}/cbits/util -I${SRCDIR}/cuda -Icbits
import "C"
