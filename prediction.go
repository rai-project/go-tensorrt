package tensorrt

import (
	"math"
	"sort"
)

type Prediction struct {
	Index       int     `json:"index"`
	Probability float32 `json:"probability"`
}

type Predictions []Prediction

// Len is the number of elements in the collection.
func (p Predictions) Len() int {
	return len(p)
}

// Less reports whether the element with
// index i should sort before the element with index j.
func (p Predictions) Less(i, j int) bool {
	pi := p[i].Probability
	pj := p[j].Probability
	return !(pi < pj || math.IsNaN(float64(pi)) && !math.IsNaN(float64(pj)))
}

// Swap swaps the elements with indexes i and j.
func (p Predictions) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

func (p Predictions) Sort() {
	sort.Sort(p)
}
