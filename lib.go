package tensorrt

// #cgo LDFLAGS: -lnvinfer -lnvcaffe_parser -lQtCore -lQtGui -lcudart
// #cgo CXXFLAGS: -std=c++11  -O3 -Wall -g
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function
// #cgo CXXFLAGS: -I/usr/include/qt4/QtGui -I/usr/include/qt4/QtCore -I/usr/include/qt4
// #cgo CXXFLAGS: -I${SRCDIR}/cbits -I${SRCDIR}/cbits/util -I${SRCDIR}/cbits/util/cuda
import "C"
