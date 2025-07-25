package cbow

import (
	"fmt"
	"io"
	"math/rand"
)

type Params struct {
	VocabSize       int     // Size of vocabulary
	EmbeddingDim    int     // Embedding dimension
	WindowSize      int     // Window size (total: left+right+1)
	NegSamples      int     // Number of negative samples
	LearningRate    float64 // Learning rate
	Epochs          int     // Number of epochs
	Seed            int64   // Seed for reproducibility
	NegativeSampler NegativeSampler
}

type BatchProvider interface {
	NextBatch() ([][]int, error) // error == io.EOF если всё
	Reset()                      // для новой эпохи
}

type Model struct {
	InputEmb  [][]float32 // VocabSize x EmbeddingDim
	OutputEmb [][]float32 // VocabSize x EmbeddingDim
	Params    Params

	rand *rand.Rand
}

func NewModel(params Params) *Model {
	rnd := rand.New(rand.NewSource(params.Seed))
	if params.NegativeSampler == nil {
		params.NegativeSampler = NewSimpleNegativeSampler(params.VocabSize, rnd)
	}

	m := &Model{
		InputEmb:  make([][]float32, params.VocabSize),
		OutputEmb: make([][]float32, params.VocabSize),
		Params:    params,
		rand:      rnd,
	}
	for i := 0; i < params.VocabSize; i++ {
		m.InputEmb[i] = make([]float32, params.EmbeddingDim)
		m.OutputEmb[i] = make([]float32, params.EmbeddingDim)
		for d := 0; d < params.EmbeddingDim; d++ {
			m.InputEmb[i][d] = float32(m.rand.NormFloat64() * 0.01) // or Uniform(-0.1, 0.1)
			m.OutputEmb[i][d] = float32(m.rand.NormFloat64() * 0.01)
		}
	}
	return m
}

func (m *Model) Embedding(wordID int) []float32 {
	return m.InputEmb[wordID]
}

func (m *Model) Train(provider BatchProvider) error {
	for epoch := 0; epoch < m.Params.Epochs; epoch++ {
		provider.Reset()
		for {
			batch, err := provider.NextBatch()
			if err == io.EOF {
				break
			}
			if err != nil {
				return fmt.Errorf("failed to read batch: %v", err)
			}
			m.trainBatch(batch)
		}
	}
	return nil
}
