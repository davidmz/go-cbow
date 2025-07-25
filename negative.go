package cbow

import (
	"math"
	"math/rand"
)

type NegativeSampler interface {
	Sample(target int) int
}

// ----------------

type simpleSampler struct {
	vocabSize int
	rand      *rand.Rand
}

func NewSimpleNegativeSampler(vocabSize int, rnd *rand.Rand) NegativeSampler {
	if rnd == nil {
		rnd = rand.New(rand.NewSource(42))
	}
	return &simpleSampler{vocabSize: vocabSize, rand: rnd}
}

func (s *simpleSampler) Sample(target int) int {
	for {
		id := s.rand.Intn(s.vocabSize)
		if id != target {
			return id
		}
	}

}

// ----------------

type unigramDistributedSampler struct {
	cdf  []float64
	rand *rand.Rand
}

// NewUnigramDistributedNegativeSampler returns a negative sampler based on unigram distribution.
// freqs - list of word frequencies (len(freqs) == vocabSize)
func NewUnigramDistributedNegativeSampler(freqs []int, rnd *rand.Rand) NegativeSampler {
	if rnd == nil {
		rnd = rand.New(rand.NewSource(42))
	}

	vocabSize := len(freqs)
	powFreqs := make([]float64, vocabSize)

	var total float64
	for i, f := range freqs {
		powFreqs[i] = math.Pow(float64(f), 0.75)
		total += powFreqs[i]
	}

	// Build Cumulative Distribution Function (CDF) in-place
	acc := 0.0
	for i, pf := range powFreqs {
		acc += pf / total
		powFreqs[i] = acc
	}

	return &unigramDistributedSampler{
		cdf:  powFreqs,
		rand: rnd,
	}
}

func (s *unigramDistributedSampler) Sample(target int) int {
	for {
		r := s.rand.Float64()
		// Binary search in cdf
		l, h := 0, len(s.cdf)-1
		for l < h {
			m := (l + h) / 2
			if r < s.cdf[m] {
				h = m
			} else {
				l = m + 1
			}
		}
		if l != target {
			return l
		}
	}
}
