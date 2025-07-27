//go:build amd64

package vmath

import (
	"fmt"
	"os"

	"golang.org/x/sys/cpu"
)

var (
	axpyImpl  func(dest, src []float32, alpha float32)
	dotImpl   func(a, b []float32) float32
	addImpl   func(dst, src []float32)
	scaleImpl func(dst []float32, alpha float32)
)

//go:noescape
func avxAxpy(dst, src []float32, alpha float32)

//go:noescape
func avxDot(a, b []float32) float32

//go:noescape
func avxAdd(dst, src []float32)

//go:noescape
func avxScale(dst []float32, alpha float32)

//go:noescape
func sse2Axpy(dst, src []float32, alpha float32)

//go:noescape
func sse2Dot(a, b []float32) float32

//go:noescape
func sse2Add(dst, src []float32)

//go:noescape
func sse2Scale(dst []float32, alpha float32)

func init() {
	switch {
	case cpu.X86.HasAVX:
		axpyImpl = avxAxpy
		dotImpl = avxDot
		addImpl = avxAdd
		scaleImpl = avxScale
	case cpu.X86.HasSSE2:
		axpyImpl = sse2Axpy
		dotImpl = sse2Dot
		addImpl = sse2Add
		scaleImpl = sse2Scale
	default:
		axpyImpl = goAxpy
		dotImpl = goDot
		addImpl = goAdd
		scaleImpl = goScale
	}
}

func Axpy(dest, src []float32, alpha float32) {
	if len(src) != len(dest) {
		fmt.Printf("length mismatch: %d != %d\n", len(src), len(dest))
		os.Exit(1)
	}

	if len(src) == 0 {
		return
	}

	axpyImpl(dest, src, alpha)
}

func Dot(a, b []float32) float32 {
	if len(a) != len(b) {
		fmt.Printf("length mismatch: %d != %d\n", len(a), len(b))
		os.Exit(1)
	}

	if len(a) == 0 {
		return 0
	}

	return dotImpl(a, b)
}

func Add(dst, src []float32) {
	if len(src) != len(dst) {
		fmt.Printf("length mismatch: %d != %d\n", len(src), len(dst))
		os.Exit(1)
	}

	if len(src) == 0 {
		return
	}

	addImpl(dst, src)
}

func Scale(dst []float32, alpha float32) {
	if len(dst) == 0 {
		return
	}

	scaleImpl(dst, alpha)
}
