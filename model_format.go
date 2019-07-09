package tensorrt

// ModelFormat ...
type ModelFormat int

const (
	ModelFormatCaffe      ModelFormat = 1
	ModelFormatOnnx       ModelFormat = 2
	ModelFormatTensorFlow ModelFormat = 3
	ModelFormatUnknown    ModelFormat = 999
)

func ClassifyModelFormat(paths ...string) ModelFormat {
	return ModelFormatCaffe
}
