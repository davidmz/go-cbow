package vmath

func goAxpy(dst, src []float32, alpha float32) {
	for i := range dst {
		dst[i] += src[i] * alpha
	}
}

func goDot(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func goAdd(dst, src []float32) {
	for i := range src {
		dst[i] += src[i]
	}
}

func goScale(dst []float32, alpha float32) {
	for i := range dst {
		dst[i] *= alpha
	}
}
