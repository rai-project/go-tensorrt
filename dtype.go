package tensorrt



// #include "cbits/predictor.hpp"
import "C"

import (
   "reflect"
)


// DType tensor scalar data type
type DType C.TensorRT_DType


const (
  UnknownType DType = C.TensorRT_Unknown
	// Byte byte tensors (go type uint8)
	Byte DType = C.TensorRT_Byte
	// Char char tensor (go type int8)
	Char DType = C.TensorRT_Char
	// Int int tensor (go type int32)
	Int DType = C.TensorRT_Int
	// Long long tensor (go type int64)
	Long DType = C.TensorRT_Long
	// Float tensor (go type float32)
	Float DType = C.TensorRT_Float
	// Double tensor  (go type float64)
	Double DType = C.TensorRT_Double
)


var types = []struct {
	typ      reflect.Type
	dataType C.TensorRT_DType
}{
	{reflect.TypeOf(uint8(0)), C.TensorRT_Byte},
	{reflect.TypeOf(int8(0)), C.TensorRT_Char},
	// {reflect.TypeOf(int16(0)), C.TensorRT_Short},
	{reflect.TypeOf(int32(0)), C.TensorRT_Int},
	{reflect.TypeOf(int64(0)), C.TensorRT_Long},
	// {reflect.TypeOf(float16(0)), C.TensorRT_Half},
	{reflect.TypeOf(float32(0)), C.TensorRT_Float},
	{reflect.TypeOf(float64(0)), C.TensorRT_Double},
}

