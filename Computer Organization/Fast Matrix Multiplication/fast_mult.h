#ifndef _H_FAST_MULT
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <float.h>

#define sub2ind(i,j,h) (i) + (j)*(h)

#define _H_FAST_MULT
#endif 

void matrixMultFast(float * const C,            /* output matrix */
                    float const * const A,      /* first matrix */
                    float const * const B,      /* second matrix */
                    int const n,                /* number of rows/cols */
                    int const ib,               /* size of i block */
                    int const jb,               /* size of j block */
                    int const kb);              /* size of k block */
