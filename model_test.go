package cbow

import (
	"io"
	"math"
	"math/rand"
	"testing"
)

type simpleProvider struct {
	batches [][][]int
	idx     int
}

func (s *simpleProvider) NextBatch() ([][]int, error) {
	if s.idx >= len(s.batches) {
		return nil, io.EOF
	}
	batch := s.batches[s.idx]
	s.idx++
	return batch, nil
}
func (s *simpleProvider) Reset() { s.idx = 0 }

func TestCBOWShapes(t *testing.T) {
	params := Params{
		VocabSize:    10,
		EmbeddingDim: 8,
		WindowSize:   3,
		NegSamples:   2,
		LearningRate: 0.05,
		Epochs:       2,
		Seed:         42,
	}
	model := NewModel(params)

	// Проверка shape
	if len(model.InputEmb) != params.VocabSize {
		t.Fatalf("InputEmb shape mismatch")
	}
	if len(model.InputEmb[0]) != params.EmbeddingDim {
		t.Fatalf("InputEmb dim mismatch")
	}
	if len(model.OutputEmb) != params.VocabSize {
		t.Fatalf("OutputEmb shape mismatch")
	}
	if len(model.OutputEmb[0]) != params.EmbeddingDim {
		t.Fatalf("OutputEmb dim mismatch")
	}
}

func TestCBOWTrainAndEmbedding(t *testing.T) {
	params := Params{
		VocabSize:    5,
		EmbeddingDim: 4,
		WindowSize:   3,
		NegSamples:   2,
		LearningRate: 0.1,
		Epochs:       3,
		Seed:         1,
	}
	model := NewModel(params)

	// Микрокорпус: "0 1 2 3 4", "4 3 2 1 0"
	batches := [][][]int{
		{{0, 1, 2, 3, 4}, {4, 3, 2, 1, 0}},
	}
	provider := &simpleProvider{batches: batches}

	// До обучения эмбеддинги почти нулевые (инициализация норм-му)
	before := make([]float32, params.EmbeddingDim)
	copy(before, model.Embedding(2))

	model.Train(provider)

	after := model.Embedding(2)
	changed := false
	for i := 0; i < params.EmbeddingDim; i++ {
		if before[i] != after[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatalf("Embeddings did not change after training")
	}

	// Проверка: длина эмбеддинга == EmbeddingDim
	if len(model.Embedding(0)) != params.EmbeddingDim {
		t.Fatalf("Embedding dim mismatch")
	}
}

func TestCBOWNegativeSample(t *testing.T) {
	params := Params{
		VocabSize:    3,
		EmbeddingDim: 2,
		WindowSize:   3,
		NegSamples:   1,
		LearningRate: 0.01,
		Epochs:       1,
		Seed:         rand.Int63(),
	}
	model := NewModel(params)
	// Проверка, что негативное сэмплирование не возвращает таргет
	target := 1
	for range 20 {
		neg := model.Params.NegativeSampler.Sample(target)
		if neg == target {
			t.Fatalf("Negative sampler returned the target")
		}
	}
}

func TestCBOWEmbeddingSimilarity(t *testing.T) {
	params := Params{
		VocabSize:    5,
		EmbeddingDim: 8,
		WindowSize:   3,
		NegSamples:   2,
		LearningRate: 0.2,
		Epochs:       100,
		Seed:         123,
	}
	model := NewModel(params)
	// Датасет: "1 2 3", "1 2 3", "1 2 3"
	batches := [][][]int{
		{{1, 2, 3}, {1, 2, 3}, {1, 2, 3}},
	}
	provider := &simpleProvider{batches: batches}

	// Косинусное расстояние
	cosDist := func(a, b []float32) float64 {
		var dot, na, nb float64
		for i := range a {
			dot += float64(a[i]) * float64(b[i])
			na += float64(a[i]) * float64(a[i])
			nb += float64(b[i]) * float64(b[i])
		}
		if na == 0 || nb == 0 {
			return 1
		}
		return 1 - dot/(math.Sqrt(na)*math.Sqrt(nb))
	}

	model.Train(provider)

	emb1 := model.Embedding(1)
	emb2 := model.Embedding(2)
	emb3 := model.Embedding(3)
	emb4 := model.Embedding(4)

	dist12 := cosDist(emb1, emb2)
	dist23 := cosDist(emb2, emb3)
	dist14 := cosDist(emb1, emb4)

	// Основная проверка: соседи должны быть ближе, чем далёкие
	if !(dist12 < dist14 && dist23 < dist14) {
		t.Fatalf("CBOW failed to cluster frequent neighbors: dist(1,2)=%.4f, dist(2,3)=%.4f, dist(1,4)=%.4f", dist12, dist23, dist14)
	}
	t.Logf("Distances after training: (1-2): %.4f, (2-3): %.4f, (1-4): %.4f", dist12, dist23, dist14)
}

func TestCBOWUpdate(t *testing.T) {
	params := Params{
		VocabSize:    3,
		EmbeddingDim: 2,
		WindowSize:   3,
		NegSamples:   1,
		LearningRate: 0.5,
		Epochs:       1,
		Seed:         77,
	}
	model := NewModel(params)

	context := []int{1, 2}
	target := 0

	before1 := make([]float32, 2)
	before2 := make([]float32, 2)
	copy(before1, model.InputEmb[1])
	copy(before2, model.InputEmb[2])

	model.trainExample(context, target)

	after1 := model.InputEmb[1]
	after2 := model.InputEmb[2]

	// Должно измениться хотя бы одно из значений
	changed := false
	for i := 0; i < 2; i++ {
		if before1[i] != after1[i] || before2[i] != after2[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatalf("Context embeddings not updated after trainExample")
	}
}

func TestCBOWEmptyContext(t *testing.T) {
	params := Params{
		VocabSize:    2,
		EmbeddingDim: 2,
		WindowSize:   1,
		NegSamples:   1,
		LearningRate: 0.1,
		Epochs:       1,
		Seed:         2,
	}
	model := NewModel(params)
	batches := [][][]int{
		{{0}}, // контекст будет пустой
	}
	provider := &simpleProvider{batches: batches}
	model.Train(provider)
	// Не должно быть паники или ошибок
}
