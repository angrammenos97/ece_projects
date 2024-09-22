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
vptree * buildvp(double *X, int n, int d);
//! Return vantage-point subtree with points inside radius
/*!
\param node A vantage-point tree
\return The vantage-point subtree
*/
vptree * getInner(vptree * T);
//! Return vantage-point subtree with points outside radius
/*!
\param node A vantage-point tree
\return The vantage-point subtree
*/
vptree * getOuter(vptree * T);
//! Return median of distances to vantage point
/*!
\param node A vantage-point tree
\return The median distance
*/
double getMD(vptree * T);
//! Return the coordinates of the vantage point
/*!
\param node A vantage-point tree
\return The coordinates [d-dimensional vector]
*/
double * getVP(vptree * T);
//! Return the index of the vantage point
/*!
\param node A vantage-point tree
\return The index to the input vector of data points
*/
int getIDX(vptree * T);
#endif
