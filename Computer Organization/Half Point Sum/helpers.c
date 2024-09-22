/*
 * Helper functions for half precision FP
 *
 * Code for functions float2half and half2float taken from
 *  http://www.mrob.com/pub/math/s10e5.h.txt
 * with slight modifications.
 * 
 * Authors 
 * ===================================================================
 * Dimitris Floros                              fcdimitr@auth.gr
 *
 * AUTh, 28 March 2017
 */

#include "helpers.h"

/*
 * Convert float to half precision
 */
half float2half(float a){

  half result = 0;
  unsigned int a_int = CAST2UINT a;
  
  int e = (a_int >> 23) & 0x000000ff; /* exponent */
  int m = (a_int & 0x007fffff);       /* mantissa */
  int s = (a_int >> 16) & 0x00008000; /* sign */

  /* unbias exponent */
  e -= 127;

  if (e == 128) {
    // infinity or NAN; preserve the leading bits of mantissa
    // because they tell whether it's a signaling or quiet NAN
    result = s | (31 << 10) | (m >> 13);
  } else if (e > 15) {
    // overflow to infinity
    result = s | (31 << 10);
  } else if (e > -15) {
    // normalized case
    if ((m & 0x00003fff) == 0x00001000) {
      // tie, round down to even
      result = s | ((e+15) << 10) | (m >> 13);
    } else {
      // all non-ties, and tie round up to even
      //   (note that a mantissa of all 1's will round up to all 0's with
      //   the exponent being increased by 1, which is exactly what we want;
      //   for example, "0.5-epsilon" rounds up to 0.5, and 65535.0 rounds
      //   up to infinity.)
      result = s | ((e+15) << 10) + ((m + 0x00001000) >> 13);
    }
  } else if (e > -25) {
    // convert to subnormal
    result = 0;
  } else {
    // zero, or underflow
    result = s;
  }

  return result;
}

/*
 * Convert half precision to float
 */
float half2float(half a_int){

  int s = a_int & 0x8000;         /* sign */
  int e = (a_int & 0x7c00) >> 10; /* exponent */
  int m = a_int & 0x03ff;         /* mantissa */

  unsigned int result = 0;
  
  s <<= 16;
  if (e == 31) {
    // infinity or NAN
    e = 255 << 23;
    m <<= 13;
    result = s | e | m;
  } else if (e > 0) {
    // normalized
    e = e + (127 - 15);
    e <<= 23;
    m <<= 13;
    result = s | e | m;
  } else if (m == 0) {
    // zero
    result = s;
  } else {
    // subnormal, value is m times 2^-24
    result = 0;
  }
  
  return CAST2FLOAT result;
}

/*
 * Print half precision in binary format
 */
void printHalf (half n){
  
  printf("\n");
  printf(" |s|  e  |     m    |\n");
  printf(" --------------------\n");
  printf(" |");
  bit2string( CAST2VOIDPTR n, 16,  1 );
  printf("|");
  bit2string( CAST2VOIDPTR n, 15,  5 );
  printf("|");
  bit2string( CAST2VOIDPTR n, 10, 10 );
  printf("|\n\n");
  
}

/*
 * Print float in binary format
 */
void printFloat (float n){

  printf("\n");
  printf(" |s|   e    |           m           |\n");
  printf(" ------------------------------------\n");
  printf(" |");
  bit2string( CAST2VOIDPTR n, 32,  1 );
  printf("|");
  bit2string( CAST2VOIDPTR n, 31,  8 );
  printf("|");
  bit2string( CAST2VOIDPTR n, 23, 23 );
  printf("|\n\n");
  
}


/*
 * bit2string: Given an unsigned number, prints the bits starting from
 * start with the given offset
 *
 * INPUTS
 *   x:   address for number to print
 *   s:   starting bit (1 or higher)
 *   len: length of bits to print (0 or greater)
 * 
 */
void bit2string (void *x, unsigned int s, unsigned int len) {
  
  unsigned int mask = 1;

  for ( ; s > 1; s-- ) mask<<=1;
  
  for ( ; len > 0; len-- ) {
    printf(mask & ( CAST2UINT *x ) ? "1" : "0");
    mask>>=1;
  }  
  
}
