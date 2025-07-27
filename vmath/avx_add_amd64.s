//go:build amd64

#include "textflag.h"

// func avxAdd(dst, src []float32)
TEXT ·avxAdd(SB), NOSPLIT, $0
    MOVQ    dst+0(FP), DI    // ptr(dst)
    MOVQ    dst+8(FP), CX    // len(dst)
    MOVQ    src+24(FP), SI   // ptr(src)
    
    // Мы уверены, что длины массивов равны и больше нуля
    
    // Количество векторных итераций (по 8 элементов)
    MOVQ    CX, BX
    SHRQ    $3, BX           // BX = len / 8
    TESTQ   BX, BX
    JZ      scalar_loop      // если < 8 элементов, идем на скалярную обработку
    
avx_loop:
    VMOVUPS (SI), Y1         // Y1 = src[i:i+8]
    VMOVUPS (DI), Y2         // Y2 = dst[i:i+8]
    VADDPS  Y1, Y2, Y2       // Y2 = dst[i:i+8] + src[i:i+8]
    VMOVUPS Y2, (DI)         // сохраняем результат обратно в dst[i:i+8]
    
    ADDQ    $32, DI          // следующие 8 элементов dst (8 * 4 байта)
    ADDQ    $32, SI          // следующие 8 элементов src
    DECQ    BX
    JNZ     avx_loop
    
scalar_loop:
    // Обрабатываем остаток (элементы, которые не поместились в векторы)
    ANDQ    $7, CX           // CX = len % 8
    TESTQ   CX, CX
    JZ      done
    
scalar_iter:
    MOVSS   (SI), X1         // X1 = src[i]
    MOVSS   (DI), X2         // X2 = dst[i]
    ADDSS   X1, X2           // X2 = dst[i] + src[i]
    MOVSS   X2, (DI)         // сохраняем результат в dst[i]
    
    ADDQ    $4, DI           // следующий элемент dst
    ADDQ    $4, SI           // следующий элемент src
    DECQ    CX
    JNZ     scalar_iter
    
done:
    VZEROUPPER               // очищаем верхние части YMM регистров
    RET

