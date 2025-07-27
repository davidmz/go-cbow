//go:build amd64

#include "textflag.h"

// func sse2Dot(a, b []float32) float32
TEXT ·sse2Dot(SB), NOSPLIT, $0
    MOVQ    a+0(FP), DI
    MOVQ    a+8(FP), CX
    MOVQ    b+24(FP), SI

    // Мы уверены, что длины массивов равны и больше нуля

    XORPS   X0, X0           // аккумулятор = 0
    
    MOVQ    CX, BX
    SHRQ    $2, BX           // векторных итераций
    TESTQ   BX, BX
    JZ      scalar_loop
    
sse_loop:
    MOVUPS  (DI), X1         // X1 = a[i:i+4]
    MOVUPS  (SI), X2         // X2 = b[i:i+4]
    MULPS   X1, X2           // X2 = a[i:i+4] * b[i:i+4]
    ADDPS   X2, X0           // X0 += X2
    
    ADDQ    $16, DI
    ADDQ    $16, SI
    DECQ    BX
    JNZ     sse_loop
    
    // Горизонтальное сложение SSE
    // X0 = [d, c, b, a]
    MOVAPS  X0, X1           // X1 = [d, c, b, a]
    SHUFPS  $0x4E, X1, X1    // X1 = [b, a, d, c] (swap high/low 64-bit)
    ADDPS   X1, X0           // X0 = [d+b, c+a, b+d, a+c]
    MOVAPS  X0, X1           // X1 = [d+b, c+a, b+d, a+c]
    SHUFPS  $0x11, X1, X1    // X1 = [c+a, d+b, c+a, d+b]
    ADDSS   X1, X0           // X0 = [*, *, *, (d+b)+(c+a)]
    
scalar_loop:
    ANDQ    $3, CX
    TESTQ   CX, CX
    JZ      done
    
scalar_iter:
    MOVSS   (DI), X1
    MOVSS   (SI), X2
    MULSS   X2, X1
    ADDSS   X1, X0
    
    ADDQ    $4, DI
    ADDQ    $4, SI
    DECQ    CX
    JNZ     scalar_iter
    
done:
    MOVSS   X0, ret+48(FP)
    RET
