//go:build amd64

#include "textflag.h"

// func avxScale(dst []float32, alpha float32)
TEXT ·avxScale(SB), NOSPLIT, $0
    MOVQ    dst+0(FP), DI    // ptr(dst)
    MOVQ    dst+8(FP), CX    // len(dst)
    MOVSS   alpha+24(FP), X0 // alpha
    
    // Мы уверены, что длина массива больше нуля
    
    // Размножаем alpha на весь YMM регистр (8 копий)
    VBROADCASTSS X0, Y0      // Y0 = [alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha]
    
    // Количество векторных итераций (по 8 элементов)
    MOVQ    CX, BX
    SHRQ    $3, BX           // BX = len / 8
    TESTQ   BX, BX
    JZ      scalar_loop      // если < 8 элементов, идем на скалярную обработку
    
avx_loop:
    VMOVUPS (DI), Y1         // Y1 = dst[i:i+8]
    VMULPS  Y0, Y1, Y1       // Y1 = dst[i:i+8] * alpha
    VMOVUPS Y1, (DI)         // сохраняем результат обратно в dst[i:i+8]
    
    ADDQ    $32, DI          // следующие 8 элементов dst (8 * 4 байта)
    DECQ    BX
    JNZ     avx_loop
    
scalar_loop:
    // Обрабатываем остаток (элементы, которые не поместились в векторы)
    ANDQ    $7, CX           // CX = len % 8
    TESTQ   CX, CX
    JZ      done
    
scalar_iter:
    MOVSS   (DI), X1         // X1 = dst[i]
    MULSS   X0, X1           // X1 = dst[i] * alpha
    MOVSS   X1, (DI)         // сохраняем результат в dst[i]
    
    ADDQ    $4, DI           // следующий элемент dst
    DECQ    CX
    JNZ     scalar_iter
    
done:
    VZEROUPPER               // очищаем верхние части YMM регистров
    RET

