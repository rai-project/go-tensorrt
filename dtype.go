
// #include "cbits/predictor.hpp"
import "C"
import "reflect"


// DType tensor scalar data type
type DType C.TensorRT_DType


const (
  UnknownType DType = C.TensoRT_Unknown
	// Byte byte tensors (go type uint8)
	Byte DType = C.TensoRT_Byte
	// Char char tensor (go type int8)
	Char DType = C.TensoRT_Char
	// Int int tensor (go type int32)
	Int DType = C.TensoRT_Int
	// Long long tensor (go type int64)
	Long DType = C.TensoRT_Long
	// Float tensor (go type float32)
	Float DType = C.TensoRT_Float
	// Double tensor  (go type float64)
	Double DType = C.TensoRT_Double
)



var types = []struct {
	typ      reflect.Type
	dataType C.TensoRT_DataType
}{
	{reflect.TypeOf(uint8(0)), C.TensoRT_Byte},
	{reflect.TypeOf(int8(0)), C.TensoRT_Char},
	// {reflect.TypeOf(int16(0)), C.TensoRT_Short},
	{reflect.TypeOf(int32(0)), C.TensoRT_Int},
	{reflect.TypeOf(int64(0)), C.TensoRT_Long},
	// {reflect.TypeOf(float16(0)), C.TensoRT_Half},
	{reflect.TypeOf(float32(0)), C.TensoRT_Float},
	{reflect.TypeOf(float64(0)), C.TensoRT_Double},
}

