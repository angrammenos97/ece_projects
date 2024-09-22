/*!
  \file   tester.c
  \brief  Validate Vantage Point implementation.

  \author Dimitris Floros
  \date   2019-10-19
*/


#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "vptree.h"

#define THRESHOLD 10e-12

/* #define VERBOSE */

static char* STR_CORRECT_WRONG[] = { "WRONG", "CORRECT" };


// =========================================================
// === STACK IMPLEMENTATION TO TEST RECUSRION INVARIANCE ===
// =========================================================

// Declare linked list node 

typedef struct node {
	vptree* T;
	int isInner;
	double md;
	struct node* link;
} node;


// Utility function to add an element data in the stack 
// insert at the beginning 
void push(node** top, vptree* T, double md, int isInner)
{
	// create new node temp and allocate memory 
	node* temp = (node*)malloc(sizeof(node));

#ifdef VERBOSE
	fprintf(stdout, "PUSH ENTERED\n");
#endif

	// check if stack (heap) is full. Then inserting an element would 
	// lead to stack overflow 
	if (!temp) {
		fprintf(stderr, "\npush: Heap Overflow\n");
		exit(1);
	}

	// initialize data into temp data field 
	temp->T = T;
	temp->isInner = isInner;
	temp->md = md;

	// put top pointer reference into temp link 
	temp->link = *top;

	// make temp as top of Stack 
	*top = temp;

}

// Remove top element from stack
void pop(node** top)
{

#ifdef VERBOSE
	fprintf(stdout, "POP ENTERED\n");
#endif

	// check for stack underflow 
	if (*top == NULL) {
		fprintf(stderr, "\npop: Stack Underflow\n");
		exit(1);
	}

	// top assign into temp 
	struct node* temp = *top;

	// return second as top
	*top = (*top)->link;

	// destroy connection between first and second 
	temp->link = NULL;

	// release memory of top node 
	free(temp);

}




// =================
// === UTILITIES ===
// =================

double dist(double* X, double* Y, int d) {
	double dist2 = 0;
	for (int i = 0; i < d; i++) {
		dist2 += (X[i] - Y[i]) * (X[i] - Y[i]);
	}
	return sqrt(dist2);
}

// Function to print all the  
// elements of the stack  
int verifyLeafPlace(node** top, double* X, int d)
{


	// check for stack underflow 
	if (*top == NULL) {
		fprintf(stderr, "\nverifyLeafPlace: Stack Underflow\n");
		exit(1);
	}

	struct node* temp = *top;

	// iterate the ancestors in stack
	while (temp != NULL) {

		// check whether point should be inside or outside
		//int isInner = dist(X, getVP(temp->T), d) <= temp->md;
		int isInner = dist(X, getVP(temp->T), d) - temp->md < THRESHOLD;

#ifdef VERBOSE
		fprintf(stdout, "%f | %f | %d == %d\n", dist(X, getVP(temp->T), d), temp->md, temp->isInner, isInner);
#endif

		// if the direction is invalid, break and return false
		if (isInner != temp->isInner) return 0;

		// assign temp link to temp 
		temp = temp->link;
	}

	return 1;

}




// ==================
// === VALIDATION ===
// ==================

int* foundInTree;

int verifyTree(vptree* T, double* vp, node** stack, double md, int isInner, int n, int d) 
{

	int isValid = 1;

	// if empty, return
	if (T == NULL) return isValid;

	int isValidAncestor = 1;
	int isLeaf = (getInner(T) == NULL && getOuter(T) == NULL);

#ifdef VERBOSE
	fprintf(stdout, "%x %x\n", stack, *stack);
#endif  

	// if leaf check ancestor path
	if (isLeaf) isValidAncestor = verifyLeafPlace(stack, getVP(T), d);


	// if inner, radius must be smaller than parent's diameter
	if (isInner && getMD(T) > 2 * md) return 0;

	// update list of indices
	int idx = getIDX(T);
	if (idx < n)
		foundInTree[idx] = 1;

	// validate distance to parent
	if (isInner)
		//isValid = dist(vp, getVP(T), d) <= md; 
		isValid = dist(vp, getVP(T), d) - md < THRESHOLD;
	else
		//isValid = dist(vp, getVP(T), d) > md; 
		isValid = dist(vp, getVP(T), d) - md > THRESHOLD;

	// recurse if not leaf
	if (!isLeaf) {

		// add to stack as inner and recurse, then pop
		push(stack, T, getMD(T), 1);
		int isValidInn = verifyTree(getInner(T), getVP(T), stack, getMD(T), 1, n, d);
		pop(stack);

		// add to stack as outer and recurse, then pop
		push(stack, T, getMD(T), 0);
		int isValidOut = verifyTree(getOuter(T), getVP(T), stack, getMD(T), 0, n, d);
		pop(stack);

		// all conditions must be true
		isValid = isValid && isValidInn && isValidOut;

	}
	else {

		// make sure ancestory path is correct
		isValid = isValid && isValidAncestor;

	}

	// return 
	return isValid;

}

int main()
{
	srand((unsigned int)time(NULL));
	int n = 100;//data
	int d = 4;//dimensions

	double* dataArr =  (double*)malloc(n * d * sizeof(double));
	double* zeros = (double*)calloc(d, sizeof(double));

	foundInTree = (int*)calloc(n, sizeof(int));

	for (int i = 0; i < n * d; i++)
		dataArr[i] = rand() % 100;

	vptree* root = buildvp(dataArr, n, d);

	node* stack = NULL;

	int isValid = verifyTree(root, zeros, &stack, DBL_MAX, 1, n, d);
	int foundAll = 1;

	assert(stack == NULL);

	for (int i = 0; i < n; i++)
		if (!foundInTree[i]) {
			foundAll = 0;
			break;
		}

	printf("Tester validation: %s PROPERTIES | %s INDICES\n",
		STR_CORRECT_WRONG[isValid], STR_CORRECT_WRONG[foundAll]);

	return 0;

}
