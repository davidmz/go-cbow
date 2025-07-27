//go:build amd64

#include "textflag.h"

// func sse2Add(dst, src []float32, alpha float32)
TEXT ·sse2Add(SB), NOSPLIT, $0
    MOVQ    dst+0(FP), DI    // ptr(dst)
    MOVQ    dst+8(FP), CX    // len(dst)
    MOVQ    src+24(FP), SI   // ptr(src)
    
    // Мы уверены, что длины массивов равны и больше нуля
    
    // Количество векторных итераций (по 4 элемента)
    MOVQ    CX, BX
    SHRQ    $2, BX           // BX = len / 4
    TESTQ   BX, BX
    JZ      scalar_loop
    
sse_loop:
    MOVUPS  (SI), X1         // X1 = src[i:i+4]
    MOVUPS  (DI), X2         // X2 = dst[i:i+4]
    ADDPS   X1, X2           // X2 = dst[i:i+4] + src[i:i+4]
    MOVUPS  X2, (DI)         // сохраняем результат
    
    ADDQ    $16, DI          // следующие 4 элемента (4 * 4 байта)
    ADDQ    $16, SI
    DECQ    BX
    JNZ     sse_loop
    
scalar_loop:
    ANDQ    $3, CX           // CX = len % 4
    TESTQ   CX, CX
    JZ      done
    
scalar_iter:
    MOVSS   (SI), X1
    MOVSS   (DI), X2
    ADDSS   X1, X2
    MOVSS   X2, (DI)
    
    ADDQ    $4, DI
    ADDQ    $4, SI
    DECQ    CX
    JNZ     scalar_iter
    
done:
    RET
