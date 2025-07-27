//go:build amd64

package vmath

import "testing"

func TestGoDot(t *testing.T) {
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
	v2 := []float32{4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}

	dot := goDot(v1, v2)
	if dot != 420.0 {
		t.Errorf("Expected dot product to be 420.0, got %f", dot)
	}
}

func TestAvxDot(t *testing.T) {
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
	v2 := []float32{4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}

	expected := goDot(v1, v2)
	dot := avxDot(v1, v2)
	if dot != expected {
		t.Errorf("Expected dot product to be %f, got %f", expected, dot)
	}
}

func TestSse2Dot(t *testing.T) {
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
	v2 := []float32{4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}

	expected := goDot(v1, v2)
	dot := sse2Dot(v1, v2)
	if dot != expected {
		t.Errorf("Expected dot product to be %f, got %f", expected, dot)
	}
}

func TestGoAxpy(t *testing.T) {
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
	v2 := []float32{4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}

	expected := []float32{9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0}
	goAxpy(v1, v2, 2.0)
	for i, v := range v1 {
		if v != expected[i] {
			t.Errorf("Expected v1[%d] to be %f, got %f", i, expected[i], v)
		}
	}
}

func TestAvxAxpy(t *testing.T) {
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
	v2 := []float32{4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}

	expected := []float32{9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0}
	avxAxpy(v1, v2, 2.0)
	for i, v := range v1 {
		if v != expected[i] {
			t.Errorf("Expected v1[%d] to be %f, got %f", i, expected[i], v)
		}
	}
}

func TestSse2Axpy(t *testing.T) {
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
	v2 := []float32{4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}

	expected := []float32{9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0}
	sse2Axpy(v1, v2, 2.0)
	for i, v := range v1 {
		if v != expected[i] {
			t.Errorf("Expected v1[%d] to be %f, got %f", i, expected[i], v)
		}
	}
}

func TestGoAdd(t *testing.T) {
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
	v2 := []float32{4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}

	expected := []float32{5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0}
	goAdd(v1, v2)
	for i, v := range v1 {
		if v != expected[i] {
			t.Errorf("Expected v1[%d] to be %f, got %f", i, expected[i], v)
		}
	}
}

func TestAvxAdd(t *testing.T) {
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
	v2 := []float32{4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}

	expected := []float32{5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0}
	avxAdd(v1, v2)
	for i, v := range v1 {
		if v != expected[i] {
			t.Errorf("Expected v1[%d] to be %f, got %f", i, expected[i], v)
		}
	}
}

func TestSse2Add(t *testing.T) {
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
	v2 := []float32{4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}

	expected := []float32{5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0}
	sse2Add(v1, v2)
	for i, v := range v1 {
		if v != expected[i] {
			t.Errorf("Expected v1[%d] to be %f, got %f", i, expected[i], v)
		}
	}
}

func TestGoScale(t *testing.T) {
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}

	expected := []float32{2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0}
	goScale(v1, 2.0)
	for i, v := range v1 {
		if v != expected[i] {
			t.Errorf("Expected v1[%d] to be %f, got %f", i, expected[i], v)
		}
	}
}

func TestAvxScale(t *testing.T) {
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}

	expected := []float32{2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0}
	avxScale(v1, 2.0)
	for i, v := range v1 {
		if v != expected[i] {
			t.Errorf("Expected v1[%d] to be %f, got %f", i, expected[i], v)
		}
	}
}

func TestSse2Scale(t *testing.T) {
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}

	expected := []float32{2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0}
	sse2Scale(v1, 2.0)
	for i, v := range v1 {
		if v != expected[i] {
			t.Errorf("Expected v1[%d] to be %f, got %f", i, expected[i], v)
		}
	}
}
