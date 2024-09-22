/*
 * Sample program for adding two half-precision floating point numbers
 * (16-bit) using integer operations.
 *
 * Authors 
 * ===================================================================
 * Dimitris Floros                              fcdimitr@auth.gr
 *
 * AUTh, 28 March 2017
 */

#include <math.h>
#include "half.h"
#include "helpers.h"

/* entry point */
int main() {

  float a, b, sum;

  half a_half, b_half, sum_half;

  printf( "Enter two numbers: " );
  scanf( "%f %f", &a, &b );     /* read input numbers */

  printf( "\n  Full precision floating points\n\n" );
  
  /* print original float numbers in binary */
  printf("Float a:\n"); printFloat(a);
  printf("Float b:\n"); printFloat(b);
  
  /* convert numbers to half */
  a_half = float2half(a);
  b_half = float2half(b);

  printf( "\n  Half precision floating points\n\n" );
  
  /* print converted numbers in binary */
  printf("Half a:\n"); printHalf(a_half);
  printf("Half b:\n"); printHalf(b_half);

  /* sum numbers in half */
  sum_half = addhalf( a_half, b_half );

  /* ground truth -- summation of floats */
  sum = a + b;

  printf( "\n  Results\n\n" );

  /* binary format of outputs */
  printf("Float sum:\n"); printFloat(sum);
  printf("Half sum:\n"); printHalf(sum_half);

  printf( "\n" );

  /* outputs in floating point format */
  printf( "float  : %f + %f = %f\n", a, b, sum );
  printf( "half   : %f + %f = %f\n",
          half2float(a_half),
          half2float(b_half),
          half2float(sum_half) );

  /* errors (maximum and relative) */

  float abs_err = fabsf(sum - half2float(sum_half));
  float rel_err = abs_err/fabsf(sum);

  /* if relative is NaN the use absolute */
  if ( isnan(rel_err) ) {
    printf( "\nAbsolute difference: %f\n\n", abs_err );

    if (abs_err < 0.01)
      printf( "\tCORRECT\n\n" );
    else
      printf( "\tERROR\n\n" );
  } else {
  
    printf( "\nRelative difference: %f\n\n", rel_err );
    if (fabsf(sum - half2float(sum_half))/fabsf(sum) < 0.01)
      printf( "\tCORRECT\n\n" );
    else
      printf( "\tERROR\n\n" );
  
    return 0;
  }
}
