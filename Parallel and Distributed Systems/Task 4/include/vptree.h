#ifndef VPTREE_H
#define VPTREE_H

typedef struct vptree {
	double *vp;
	double md;
	int idx;
	struct vptree *inner;
	struct vptree *outer;
}vptree;

// type definition of vptree
// ========== LIST OF ACCESSORS
//! Build vantage-point tree given input dataset X
/*!
\param X Input data points, stored as [n-by-d] array
\param n Number of data points (rows of X)
\param d Number of dimensions (columns of X)
\return The vantage-point tree
*/
vptree* buildvp_cpu(double* X, int n, int d);
vptree* buildvp_gpu(double* X, int n, int d);
static vptree* buildvp(double* X, int n, int d) { return buildvp_gpu(X, n, d); }
//! Return vantage-point subtree with points inside radius
/*!
\param node A vantage-point tree
\return The vantage-point subtree
*/
static vptree* getInner(vptree* T) { return T->inner; }
//! Return vantage-point subtree with points outside radius
/*!
\param node A vantage-point tree
\return The vantage-point subtree
*/
static vptree* getOuter(vptree* T) { return T->outer; }
//! Return median of distances to vantage point
/*!
\param node A vantage-point tree
\return The median distance
*/
static double getMD(vptree* T) { return T->md; }
//! Return the coordinates of the vantage point
/*!
\param node A vantage-point tree
\return The coordinates [d-dimensional vector]
*/
static double* getVP(vptree* T) { return T->vp; }
//! Return the index of the vantage point
/*!
\param node A vantage-point tree
\return The index to the input vector of data points
*/
static int getIDX(vptree* T) { return T->idx; }

#endif
