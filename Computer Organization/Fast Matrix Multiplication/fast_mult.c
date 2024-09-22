/*
 * NOTES
 *
 *   For okeanos best blocks for size 512x512 are [128 64 8] with a
 *   speedup of 10.
 *
 */

#include "fast_mult.h"

void matrixMultFast(float * const C,            /* output matrix */
                    float const * const A,      /* first matrix */
                    float const * const B,      /* second matrix */
                    int const n,                /* number of rows/cols */
                    int const ib,               /* size of i block */
                    int const jb,               /* size of j block */
                    int const kb)               /* size of k block */
{


    /* <YOUR-CODE-HERE> */

    float acc00 , acc01 , acc10 , acc11;
    int ii , kk , j , i , k;

    for (ii = 0; ii < n; ii += ib)
    {
        for (kk = 0; kk < n; kk += kb)
        {
            for (j=0; j < n; j += 2)
            {
                for (i = ii; i < ii + ib; i += 2 )
                {
                    if (kk == 0)
                        acc00 = acc01 = acc10 = acc11 = 0;
                    else
                    {
                        acc00 = C[ sub2ind(i,j,n) ];
                        acc01 = C[ sub2ind(i,j+1,n) ];
                        acc10 = C[ sub2ind(i+1,j,n) ];
                        acc11 = C[ sub2ind(i+1,j+1,n) ];
                    }
                    for (k = kk; k < kk + kb; k++)
                    {
                        acc00 += A[ sub2ind(i,k,n) ] * B[ sub2ind(k,j,n) ];
                        acc01 += A[ sub2ind(i,k,n) ] * B[ sub2ind(k,j+1,n) ];
                        acc10 += A[ sub2ind(i+1,k,n) ] * B[ sub2ind(k,j,n) ];
                        acc11 += A[ sub2ind(i+1,k,n) ] * B[ sub2ind(k,j+1,n) ];
                    }
                    C[ sub2ind(i,j,n) ] = acc00;
                    C[ sub2ind(i,j+1,n) ] = acc01;
                    C[ sub2ind(i+1,j,n) ] = acc10;
                    C[ sub2ind(i+1,j+1,n) ] = acc11;
                }
            }
        }
    }

} // end function 'matrixMultFast4'
