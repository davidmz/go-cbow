//go:build amd64

#include "textflag.h"

// func avxAxpy(dst, src []float32, alpha float32)
TEXT ·avxAxpy(SB), NOSPLIT, $0
    MOVQ    dst+0(FP), DI    // ptr(dst)
    MOVQ    dst+8(FP), CX    // len(dst)
    MOVQ    src+24(FP), SI   // ptr(src)
    MOVSS   alpha+48(FP), X0 // alpha
    
    // Мы уверены, что длины массивов равны и больше нуля
    
    // Размножаем alpha на весь YMM регистр (8 копий)
    VBROADCASTSS X0, Y0      // Y0 = [alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha]
    
    // Количество векторных итераций (по 8 элементов)
    MOVQ    CX, BX
    SHRQ    $3, BX           // BX = len / 8
    TESTQ   BX, BX
    JZ      scalar_loop      // если < 8 элементов, идем на скалярную обработку
    
avx_loop:
    VMOVUPS (SI), Y1         // Y1 = src[i:i+8]
    VMULPS  Y0, Y1, Y1       // Y1 = src[i:i+8] * alpha
    VMOVUPS (DI), Y2         // Y2 = dst[i:i+8]
    VADDPS  Y1, Y2, Y2       // Y2 = dst[i:i+8] + (src[i:i+8] * alpha)
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
    MULSS   X0, X1           // X1 = src[i] * alpha
    MOVSS   (DI), X2         // X2 = dst[i]
    ADDSS   X1, X2           // X2 = dst[i] + (src[i] * alpha)
    MOVSS   X2, (DI)         // сохраняем результат в dst[i]
    
    ADDQ    $4, DI           // следующий элемент dst
    ADDQ    $4, SI           // следующий элемент src
    DECQ    CX
    JNZ     scalar_iter
    
done:
    VZEROUPPER               // очищаем верхние части YMM регистров
    RET

