#ifdef DEBUG
#include <assert.h>
#include <stdio.h>
#endif

int leaf (int g, int h, int i, int j) {

  int f;
  
  f = (g + h) - (i + j);
  
  return f;

}


void main() {


  int g = 2;
  int h = 3;
  int i = 4;
  int j = 5;

  int f = leaf( g, h, i, j );
  
#ifdef DEBUG
  
  assert( f == -4 );
  printf( "Assertion passed\n" );
  
#endif
  

}
