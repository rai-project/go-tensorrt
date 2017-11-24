// +build !linux

package tensorrt

import "C"
import (
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework/framework/options"
)

var (
	invalidSystemError = errors.New("invalid system. TensorRT is only available on linux systems")
)

type Predictor struct {
}

func New(opts0 ...options.Option) (*Predictor, error) {
	return nil, invalidSystemError
}

func (p *Predictor) Predict(input []float32) (Predictions, error) {
	return nil, invalidSystemError
}

func (p *Predictor) StartProfiling(name, metadata string) error {
	return invalidSystemError
}

func (p *Predictor) EndProfiling() error {
	return invalidSystemError
}

func (p *Predictor) DisableProfiling() error {
	return invalidSystemError
}

func (p *Predictor) ReadProfile() (string, error) {
	return "", invalidSystemError
}

func (p Predictor) Close() {

}
