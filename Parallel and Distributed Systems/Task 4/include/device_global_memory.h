#ifndef DEVICE_GLOBAL_MEMORY_H_
#define DEVICE_GLOBAL_MEMORY_H_

#define MAXTHREADSPERBLOCK	32	//maximun threads per block

/*This header has all pointers to device global memory from the host*/

//vptree.cu global variables
extern double* d_points;
extern double* d_pointsAux;
extern unsigned int* d_indexes;
extern unsigned int* d_indexesAux;
extern unsigned int* d_vpSwaps;
extern double* d_treeMDs;
extern unsigned int* d_treeIDXs;
extern unsigned int* d_nodesOffset;
extern unsigned int* d_nodesLength;

//distances.cu global variables
extern double* d_distances;

//quick_select.cu global variables
extern double* d_qsAux;
extern unsigned int* d_f;
extern unsigned int* d_t;
extern unsigned int* d_addr;
extern unsigned int* d_NFs;
extern char* d_e;

//knn_search.cu global variables
extern double* d_qpoints;
extern double* d_ndist;
extern unsigned int* d_nidx;
extern unsigned int* d_offsetsStack;
extern unsigned int* d_lengthsStack;
extern double* d_parentNDistStack;
extern double* d_parentMdStack;
extern char* d_isInnerStack;

//functions to initialize global memory
int qs_memory_allocate(unsigned int numberOfPoints, unsigned int maxParallelNodes);

int di_memory_allocate(unsigned int numberOfPoints);

int vp_memory_allocate(unsigned int numberOfPoints, unsigned int dimensionOfPoints);

int knn_memory_allocate(unsigned int n, unsigned int m, unsigned int d, unsigned int k);

//functions to free global memory
void qs_memory_deallocate();

void di_memory_deallocate();

void vp_memory_deallocate();

void knn_deallocate();

#endif // !DEVICE_GLOBAL_MEMORY_H_
