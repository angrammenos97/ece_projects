#ifndef TESTER_HELPER_H
#define TESTER_HELPER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "knnring.h"

// ============================== ACCESS MACROS

// ~~~~~~~~~~~~~~~~~~~~ COL-MAJOR

#define COLMAJOR 0
#define idx_cm(i,j,m,n) (i) + (j)*(m)

#define corpus_cm(i,j)         corpus          [idx_cm(i,j,n,d)               ]
#define corpusAll_cm(i,j)      corpusAll       [idx_cm(i,j,n*p,d)             ]
#define knnresnidx_cm(i,j)     knnres.nidx     [idx_cm(i,j,knnres.m,knnres.k) ]
#define knnresndist_cm(i,j)    knnres.ndist    [idx_cm(i,j,knnres.m,knnres.k) ]
#define knnresallnidx_cm(i,j)  knnresall.nidx  [idx_cm(i,j,n*p,k)             ]
#define knnresallndist_cm(i,j) knnresall.ndist [idx_cm(i,j,n*p,k)             ]

#define X_cm(i,j) X[idx_cm(i,j,n,d)]
#define Y_cm(i,j) Y[idx_cm(i,j,m,d)]

// ~~~~~~~~~~~~~~~~~~~~ ROW-MAJOR

#define ROWMAJOR 1
#define idx_rm(i,j,m,n) (j) + (i)*(n)

#define corpus_rm(i,j)         corpus          [idx_rm(i,j,n,d)               ]
#define corpusAll_rm(i,j)      corpusAll       [idx_rm(i,j,n*p,d)             ]
#define knnresnidx_rm(i,j)     knnres.nidx     [idx_rm(i,j,knnres.m,knnres.k) ]
#define knnresndist_rm(i,j)    knnres.ndist    [idx_rm(i,j,knnres.m,knnres.k) ]
#define knnresallnidx_rm(i,j)  knnresall.nidx  [idx_rm(i,j,n*p,k)             ]
#define knnresallndist_rm(i,j) knnresall.ndist [idx_rm(i,j,n*p,k)             ]

#define X_rm(i,j) X[idx_rm(i,j,n,d)]
#define Y_rm(i,j) Y[idx_rm(i,j,m,d)]

static char * STR_CORRECT_WRONG[] = {"WRONG", "CORRECT"};

// =================
// === UTILITIES ===
// =================

double distColMajor(double const * const X, double const * const Y,
                    int i, int j,
                    int d, int n, int m){

  /* compute distance */
  double dist = 0;
  for (int l = 0; l < d; l++){
    dist += ( X_cm(i,l) - Y_cm(j,l) ) * ( X_cm(i,l) - Y_cm(j,l) );
  }

  return sqrt(dist);
}

double distRowMajor(double const * const X, double const * const Y,
                    int i, int j,
                    int d, int n, int m){

  /* compute distance */
  double dist = 0;
  for (int l = 0; l < d; l++){
    dist += ( X_rm(i,l) - Y_rm(j,l) ) * ( X_rm(i,l) - Y_rm(j,l) );
  }

  return sqrt(dist);
}



// ==================
// === VALIDATION ===
// ==================

//! kNN validator
/*!
   The function asserts correctness of the kNN results by:
     (i)   Checking that reported distances are correct
     (ii)  Validating that distances are sorted in non-decreasing order
     (iii) Ensuring there are no other points closer than the kth neighbor
*/
int validateResult( knnresult knnres,
                    double const * const corpus, double const * const query,
                    int n, int m, int d, int k, int ap ) {

  /* loop through all query points */
  for (int j = 0; j < m; j++ ){

    /* max distance so far (equal to kth neighbor after nested loop) */
    double maxDist = -1;

    /* mark all distances as not computed */
    int * visited = (int *) calloc( n, sizeof(int) );

    /* go over reported k nearest neighbors */
    for (int l = 0; l < k; l++ ){

      if (ap == COLMAJOR){

        /* keep list of visited neighbors */
        visited[ knnresnidx_cm(j,l) ] = 1;

        /* get distance to stored index */
        double distxy = distColMajor( corpus, query, knnresnidx_cm(j,l), j, d, n, m );

        /* make sure reported distance is correct */
        if ( fabs( knnresndist_cm(j,l) - distxy ) > 1e-6 ) return 0;

        /* distances should be non-decreasing */
        if ( knnresndist_cm(j,l) < maxDist ) return 0;

        /* update max neighbor distance */
        maxDist = knnresndist_cm(j,l);

      } else {

        /* keep list of visited neighbors */
        visited[ knnresnidx_rm(j,l) ] = 1;

        /* get distance to stored index */
        double distxy = distRowMajor( corpus, query, knnresnidx_rm(j,l), j, d, n, m );

        /* make sure reported distance is correct */
        if ( fabs( knnresndist_rm(j,l) - distxy ) > 1e-6 ) return 0;

        /* distances should be non-decreasing */
        if ( knnresndist_rm(j,l) < maxDist ) return 0;

        /* update max neighbor distance */
        maxDist = knnresndist_rm(j,l);


      }

    } /* for (k) -- reported nearest neighbors */

    /* now maxDist should have distance to kth neighbor */

    /* check all un-visited points */
    for (int i = 0; i < n; i++ ){

      /* check only (n-k) non-visited nodes */
      if (!visited[i]){

        double distxy;

        /* get distance to unvisited vertex */
        if (ap == COLMAJOR)
          distxy = distColMajor( corpus, query, i, j, d, n, m );
        else
          distxy = distRowMajor( corpus, query, i, j, d, n, m );

        /* point cannot be closer than kth distance */
        if ( distxy < maxDist ) return 0;

      } /* if (!visited[i]) */

    } /* for (i) -- unvisited notes */

    /* deallocate memory */
    free( visited );

  } /* for (j) -- query points */

  /* return */
  return 1;

}


#endif
