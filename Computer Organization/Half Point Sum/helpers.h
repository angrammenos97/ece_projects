/*
 * Helper functions for half precision FP
 *
 * Authors 
 * ===================================================================
 * Dimitris Floros                              fcdimitr@auth.gr
 *
 * AUTh, 28 March 2017
 */
#ifndef HELPERS_H
#define HELPERS_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define CAST2FLOAT    *(float *) &
#define CAST2UINT     *(unsigned int *) &
#define CAST2VOIDPTR  (void *) &

/* custom type */
typedef unsigned short half;

half  float2half(float a); /* convert from float to half */
float half2float(half  a); /* convert from half to float */
void  printFloat(float n); /* print binary format of half */
void  printHalf (half  n); /* print binary format of float */

/* convert bit 2 string */
void bit2string(void *x, unsigned int s, unsigned int off);

#endif
