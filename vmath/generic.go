//go:build !amd64

package vmath

func Axpy(dst, src []float32, alpha float32) { goAxpy(dst, src, alpha) }
func Dot(a, b []float32) float32             { return goDot(a, b) }
func Add(dst, src []float32)                 { goAdd(dst, src) }
