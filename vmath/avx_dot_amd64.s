//go:build amd64

#include "textflag.h"

// func avxDot(a, b []float32) float32
TEXT ·avxDot(SB), NOSPLIT, $0
    MOVQ    a+0(FP), DI   // ptr(a)
    MOVQ    a+8(FP), CX   // len(a)
    MOVQ    b+24(FP), SI  // ptr(b)

    // Мы уверены, что длины массивов равны и больше нуля

    // Обнуляем AVX аккумулятор (8 float32)
    VXORPS  Y0, Y0, Y0
    
    // Количество векторных итераций (по 8 элементов)
    MOVQ    CX, BX
    SHRQ    $3, BX        // BX = len / 8
    TESTQ   BX, BX
    JZ      scalar_loop   // если < 8 элементов, идем на скалярную обработку
    
avx_loop:
    VMOVUPS (DI), Y1      // Y1 = a[i:i+8]
    VMOVUPS (SI), Y2      // Y2 = b[i:i+8]
    VMULPS  Y1, Y2, Y1    // Y1 = a[i:i+8] * b[i:i+8]
    VADDPS  Y0, Y1, Y0    // Y0 += Y1 (аккумулируем)
    
    ADDQ    $32, DI       // следующие 8 элементов a (8 * 4 байта)
    ADDQ    $32, SI       // следующие 8 элементов b
    DECQ    BX
    JNZ     avx_loop
    
    // Горизонтальное сложение Y0 в один скаляр
    // Y0 = [a7, a6, a5, a4, a3, a2, a1, a0]
    VEXTRACTF128 $1, Y0, X1    // X1 = верхние 4 элемента [a7, a6, a5, a4]
    VADDPS       X0, X1, X0    // X0 = [a7+a3, a6+a2, a5+a1, a4+a0]
    VHADDPS      X0, X0, X0    // X0 = [*, *, a7+a6+a5+a4+a3+a2+a1+a0, *]
    VHADDPS      X0, X0, X0    // X0 = [*, *, *, сумма_всех]
    
scalar_loop:
    // Обрабатываем остаток (элементы, которые не поместились в векторы)
    ANDQ    $7, CX        // CX = len % 8
    TESTQ   CX, CX
    JZ      done
    
scalar_iter:
    MOVSS   (DI), X1      // X1 = a[i]
    MOVSS   (SI), X2      // X2 = b[i]
    MULSS   X2, X1        // X1 = a[i] * b[i]
    ADDSS   X1, X0        // X0 += a[i] * b[i]
    
    ADDQ    $4, DI        // следующий элемент a
    ADDQ    $4, SI        // следующий элемент b
    DECQ    CX
    JNZ     scalar_iter
    
done:
    MOVSS   X0, ret+48(FP) // возвращаем результат
    VZEROUPPER            // очищаем верхние части YMM регистров
    RET

