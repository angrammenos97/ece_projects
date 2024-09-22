/*
 * Library containing function implementing addition of two
 * half-precision numbers in C.
 * 
 * Author 
 * ===================================================================
 * (name)                                       (email)
 *
 * AUTh, (date)       
 */

#include "helpers.h"

/*
 * addhalf: Add two half-precision floating point numbers using
 * integer operations.
 *
 * INPUTS
 *   a: address of first number in half
 *   b: address of second number in half
 *
 * OUTPUT
 *   sum: The result of the addition of the two numbers in half.
 *
 * NOTE
 *   This function only implements base case scenario. It does not
 *   take into account corner cases. It should include the cases when
 *   the output is equal to zero or both inputs are equal to zero.
 *
 *   Validate following inputs:
 *      0.0 +  0.0 =  0.0
 *      1.0 + -1.0 =  0.0
 *      0.1 +  0.0 =  0.1
 *     -0.1 +  0.0 = -0.1 
 */
half addhalf( const half a, const half b ) {

  half sum = 0;

  /* <<YOUR-CODE-HERE>> */

  return sum;
  
}
