#ifndef MUL_ARR_H
#define MUL_ARR_H

#include <ap_int.h>

typedef ap_uint<8> uint8;
typedef ap_uint<32> uint32;

#define lm 7
#define ln 7
#define lp 7

#define n  (1 << ln)
#define m  (1 << lm)
#define p  (1 << lp)

void mulArrHW(uint8 A[n][m], uint8 B[m][p], uint32 C[n][p]);

#endif
