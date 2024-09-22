/* **********************************************************************
 *
 * MATRIX-MULT
 * ----------------------------------------------------------------------
 *
 *   Simple program to showcase the benefit of cache blocking.
 *
 * NOTES
 *
 *   Matrices are given as 1-D arrays.
 *
 *
 * ********************************************************************** */

#include "fast_mult.h"

#define MAX_ITER 2
#define MIN_SIZE 512
#define MAX_SIZE 512


/*
 * matrixMult - Matrix multiplication
 */
void matrixMult(float * const C,            /* output matrix */
                float const * const A,      /* first matrix */
                float const * const B,      /* second matrix */
                int const n) {              /* number of rows/cols */
    int i,j,k;
  for (i = 0; i < n; i++) {             /* rows */
    for (j = 0; j < n; j++) {           /* cols */

      /* initialize output value */
      C[ sub2ind(i,j,n) ] = 0;

      for (k = 0; k < n; k++) {         /* accumulate products */
        C[ sub2ind(i,j,n) ] +=
          A[ sub2ind(i,k,n) ] * B[ sub2ind(k,j,n) ];
      }

    }
  }

} // end function 'matrixMult'


/*
 * matrixInitAdd - Initialize matrix by summing indices
 */
void matrixInitAdd(float * const M,        /* matrix pointer */
                   int const n) {          /* number of rows/cols */

    int i,j;
  for (i = 0; i < n; i++)              /* rows */
    for (j = 0; j < n; j++)            /* cols */
      M[ sub2ind(i,j,n) ] = i + j;

} // end function 'matrixInitAdd'


/*
 * matrixInitSub - Initialize matrix by substracting indices
 */
void matrixInitSub(float * const M,        /* matrix pointer */
                   int const n) {          /* number of rows/cols */

    int i,j;
  for (i = 0; i < n; i++)              /* rows */
    for (j = 0; j < n; j++)            /* cols */
      M[ sub2ind(i,j,n) ] = i - j;

} // end function 'matrixInitSub'

int main(int argc, char **argv) {


    int n;
  for ( n = MIN_SIZE; n <= MAX_SIZE; n*=2 ) {

    float *A, *B, *C, *D;         /* matrix declarations */

    struct timeval start, end;    /* time structs */
    double timeNorm = 0;          /* execution time in ms */

    printf("...checking size %dx%d...\n\n", n, n);

    /* allocate matrices */
    A = (float *) malloc( n*n*sizeof(float) );
    B = (float *) malloc( n*n*sizeof(float) );
    C = (float *) malloc( n*n*sizeof(float) );
    D = (float *) malloc( n*n*sizeof(float) );

    /* initialize matrices */
    matrixInitAdd( A, n );      /* A(i,j) = i+j */
    matrixInitSub( B, n );      /* B(i,j) = i-j */

    /* compute normal matrix multiplication */
    int it;
    for (it = 0; it < MAX_ITER; it++) {
      gettimeofday(&start, NULL);
      matrixMult( C, A, B, n );
      gettimeofday(&end, NULL);

      if (it > 0)  /* drop first */
        timeNorm += ( (end.tv_sec - start.tv_sec) * 1000.0 +    /* sec to ms */
                      (end.tv_usec - start.tv_usec) / 1000.0 ); /* us to ms */
    }

    timeNorm = timeNorm / (MAX_ITER-1);


    double timeFast = 0;

    /* compute fast matrix multiplication */
    for (it = 0; it < MAX_ITER; it++) {
      gettimeofday(&start, NULL);
      matrixMultFast( D, A, B, n, 128, 64, 8 );
      gettimeofday(&end, NULL);

      if (it > 0)  /* drop first */
        timeFast += ( (end.tv_sec - start.tv_sec) * 1000.0 +
                      (end.tv_usec - start.tv_usec) / 1000.0 );

    }

    timeFast = timeFast / (MAX_ITER-1);

#ifdef FLAG_DEBUG

    float res = 0;

    for (int i = 0; i < n*n; i++)
      res += fabsf( C[i] - D[i] );

    assert( res == 0 );

    printf( "\n" );
    printf( "  Validation        PASS\n" );

#else

    float res = C[n] - D[n];
    assert( res == 0 );

#endif

    double speedup = timeNorm/timeFast;

    double grade = (10.0 / 9.0) * (speedup - 1);

    /* geq 0 */
    grade = fmax( grade, 0 );

    /* leq 10 */
    grade = fmin( grade, 10 );

    /* output times */
    printf( "\n" );
    printf( "  Normal   %7.2f ms\n", timeNorm );
    printf( "  Faster   %7.2f ms\n", timeFast );

    printf( "\n" );

    printf( "  Speed-up %7.2f\n", speedup );

    printf( "\n" );

    if (speedup > 1.8)
      printf( "  Acceleration      PASS\n" );
    else
      printf( "  Acceleration      FAIL\n" );

    printf( "\n" );

    printf( "  Grade              %3.1f\n", grade );

    printf( "\n" );

  }
  return 0;

}

/* **********************************************************************
 *
 * AUTHORS
 *
 *   Dimitris Floros                         fcdimitr@auth.gr
 *
 * VERSION
 *
 *   0.1 - May 09, 2017
 *
 * CHANGELOG
 *
 *   0.1 (May 09, 2017) - Dimitris
 *       * initial implementation
 *
 * ********************************************************************** */
