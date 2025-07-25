package cbow

import (
	"math"
)

func (m *Model) trainBatch(batch [][]int) {
	for _, sentence := range batch {
		for pos, word := range sentence {
			context := make([]int, 0, m.Params.WindowSize-1)
			// Собираем контекст
			for i := -m.Params.WindowSize / 2; i <= m.Params.WindowSize/2; i++ {
				if i == 0 {
					continue
				}
				idx := pos + i
				if idx >= 0 && idx < len(sentence) {
					context = append(context, sentence[idx])
				}
			}
			if len(context) == 0 {
				continue
			}
			m.trainExample(context, word)
		}
	}
}

func (m *Model) trainExample(context []int, target int) {
	// 1. Считаем средний контекстный вектор
	embDim := m.Params.EmbeddingDim
	contextVec := make([]float32, embDim)
	for _, wid := range context {
		for d := 0; d < embDim; d++ {
			contextVec[d] += m.InputEmb[wid][d]
		}
	}
	for d := 0; d < embDim; d++ {
		contextVec[d] /= float32(len(context))
	}

	// 2. Обновляем по позитивному примеру (context -> target)
	m.update(context, contextVec, target, 1)

	// 3. Негативное сэмплирование
	for i := 0; i < m.Params.NegSamples; i++ {
		negTarget := m.Params.NegativeSampler.Sample(target)
		m.update(context, contextVec, negTarget, 0)
	}

	// 4. Обновляем контекстные эмбеддинги
	// (Если нужно, можно аккумулировать градиенты и обновлять в конце)
}

func (m *Model) update(context []int, contextVec []float32, target int, label int) {
	var score float32
	for d := 0; d < m.Params.EmbeddingDim; d++ {
		score += contextVec[d] * m.OutputEmb[target][d]
	}
	pred := sigmoid(score)
	grad := float32(label) - pred
	lr := float32(m.Params.LearningRate)

	// Update output embedding
	for d := 0; d < m.Params.EmbeddingDim; d++ {
		m.OutputEmb[target][d] += lr * grad * contextVec[d]
	}
	// Update input embeddings (распределяем градиент по каждому контекстному слову)
	for _, wid := range context {
		for d := 0; d < m.Params.EmbeddingDim; d++ {
			m.InputEmb[wid][d] += lr * grad * m.OutputEmb[target][d] / float32(len(context))
		}
	}
}

func sigmoid(x float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-x))))
}
