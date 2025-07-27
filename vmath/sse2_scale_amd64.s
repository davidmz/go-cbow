//go:build amd64

#include "textflag.h"

// func sse2Scale(dst []float32, alpha float32)
TEXT ·sse2Scale(SB), NOSPLIT, $0
    MOVQ    dst+0(FP), DI    // ptr(dst)
    MOVQ    dst+8(FP), CX    // len(dst)
    MOVSS   alpha+24(FP), X0 // alpha
    
    // Размножаем alpha на весь XMM регистр (4 копии)
    SHUFPS  $0, X0, X0       // X0 = [alpha, alpha, alpha, alpha]
    
    // Количество векторных итераций (по 4 элемента)
    MOVQ    CX, BX
    SHRQ    $2, BX           // BX = len / 4
    TESTQ   BX, BX
    JZ      scalar_loop
    
sse_loop:
    MOVUPS  (DI), X1         // X1 = dst[i:i+4]
    MULPS   X0, X1           // X1 = dst[i:i+4] * alpha
    MOVUPS  X1, (DI)         // сохраняем результат
    
    ADDQ    $16, DI          // следующие 4 элемента (4 * 4 байта)
    DECQ    BX
    JNZ     sse_loop
    
scalar_loop:
    ANDQ    $3, CX           // CX = len % 4
    TESTQ   CX, CX
    JZ      done
    
scalar_iter:
    MOVSS   (DI), X1
    MULSS   X0, X1
    MOVSS   X1, (DI)
    
    ADDQ    $4, DI
    DECQ    CX
    JNZ     scalar_iter
    
done:
    RET
