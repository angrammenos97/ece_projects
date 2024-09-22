//////////////// Ising Code Here ////////////////
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define WeightMatDim 5	// Weight Matrix Dimension
#define FloatError 1e-6	// Float error
#define TileSize 32		// Size of tiles partitioning the matrix - each tile calculates TileSize x TileSize moments
#define NumberOfRows 8	// Rows of each block of threads - each block is of size NumberOfRows x TileSi
#define RowsHelping (NumberOfRows + 4)	// Rows of global memory to be loaded into shared (for the exercise NumberOfRows + 4)
#define ColumnsHelping (TileSize + 4)	// Columns of global memory to be loaded into shared (for the exercise TileSize + 4)


__global__ void calculateFrameShared(int* G_d, int* GNext_d, double* w_d, int n, int *same_matrix_d)
{
	__shared__ double w_s[25]; // Shared matrix to hold weight table and accelerate access
	__shared__ int g_s[RowsHelping][ColumnsHelping]; // Shared matrix to hold moments needed for the current iteration

	// Copy w_d matrix from global to shared memory with optimal access of the former
	// as far as warps are concerned - indexing used is based only on block dims

	if ((threadIdx.x < 25) && (threadIdx.y == 0)) // Use only 25 initial threads of the first warp
		w_s[threadIdx.x] = w_d[threadIdx.x];

	/* Initialize thread coordinates to be used for global memory access*/
	int x = blockIdx.x * TileSize + threadIdx.x;
	int y = blockIdx.y * TileSize + threadIdx.y;

	// Do as many time needed to cover the whole area assigned to the block
	// Copy block elements for current iteration from global memory to shared in a warp
	// oriented manner copying a whole block of size [(NumberOfRows+4) x (TileSize+4)] in order to 
	// transfer all elements accessed during the iteration and thus gain in speed of execution

	for (int j = 0; j < TileSize; j += NumberOfRows) { // for every block chunk of tile in the y axis
		int counterY = 0; // variable to help with accessing rows of G_d table for copying
		// Copying moments for iteration j into shared memory
		for (int helperY = threadIdx.y; helperY < RowsHelping; helperY += NumberOfRows) { // for every row of the subtile to be loaded 		
			int counterX = 0; // variable to help with accessing columns of G_d table for copying
			int rowNumber = (y - 2 + counterY * NumberOfRows + n) % n; // Current Row of G_d to be dereferenced
			for (int helperX = threadIdx.x; helperX < ColumnsHelping; helperX += TileSize) {
				int columnNumber = (x - 2 + counterX * TileSize + n) % n; // Current Column of G_d to be dereferenced				
				g_s[helperY][helperX] = G_d[rowNumber*n + columnNumber]; // Dereference G_d and copy to shared memory
				++counterX; // increase helper counter for columns
			}
			++counterY; // increase helper counter for rows
		}
		__syncthreads(); // waiting for all threads to be done

		// Evaluating moments in subblock through g_s and w_s matrices
		if (x < n) {  // Check whether thread x coordinate is out of bounds
			if (y < n) { // Check whether thread y coordinate is out of bounds
				double influence = 0.0; // weighted influence of neighbors
				for (int i = -2; i <= 2; i++) {	// for every row of weight matrix
					int r = threadIdx.y + 2 + i;	// add 2 to y coordinate as is in block manner to "center" block in g_s dereferencing
					for (int t = -2; t <= 2; t++) {	// for every weight of a row in weight matrix
						int c = threadIdx.x + 2 + t;	// add 2 to x coordinate as is in block manner to "center" block in g_s dereferencing
						influence += g_s[r][c] * w_s[(i + 2) * WeightMatDim + (t + 2)];	// +2 cause of the i and t offset
					}// for t < WeightMatDim
				}// for i < WeightMatDim

				/*Update state for current point*/
				if (influence > FloatError)			// apply threshold for floating point error
					*(GNext_d + y * n + x) = 1;
				else if (influence < -FloatError)	// apply threshold for floating point error
					*(GNext_d + y * n + x) = -1;
				else								// stay the same
					*(GNext_d + y * n + x) = g_s[threadIdx.y + 2][threadIdx.x + 2];

				// Did the magnetic moment changed?
				if (*(GNext_d + y * n + x) != g_s[threadIdx.y + 2][threadIdx.x + 2])
					*same_matrix_d = 0;
			} // if (y < n)
		} // if (x < n)

		y += NumberOfRows; // Update y coordinate as we move down the tile
		__syncthreads(); // Synchronize threads for next iterartion
	} // for (j < TileSize)
}

void ising(int* G, double* w, int k, int n)
{
	/*Declare and initialize memory to use on device*/
	int* G_d, *GNext_d; double* w_d; // Pointers to use for matrix store on device
	int *same_matrix_d;	// device parameter to hold if iterations should proceed
	int same_matrix = 1;	// corresponding host parameter

	cudaMalloc((void**)&G_d, n * n * sizeof(int));
	cudaMalloc((void**)&GNext_d, n * n * sizeof(int));
	cudaMalloc((void**)&w_d, WeightMatDim * WeightMatDim * sizeof(double));
	cudaMalloc((void**)&same_matrix_d, sizeof(int));

	/*Create streams to overlap data transfers and gain in speed*/
	cudaStream_t stream1;
	cudaStream_t stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	// Copy data
	cudaMemcpyAsync(G_d, G, n * n * sizeof(int), cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(GNext_d, G, n * n * sizeof(int), cudaMemcpyHostToDevice, stream2);
	cudaMemcpy(w_d, w, WeightMatDim * WeightMatDim * sizeof(double), cudaMemcpyHostToDevice);

	// Destroy streams
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

	/*Declare grid and block sizes and compensate for matrix not divided with block size*/
	dim3 dimBlock(TileSize, NumberOfRows);
	dim3 dimGrid((n + TileSize - 1) / TileSize, (n + TileSize - 1) / TileSize);

	/*--------------------------------------------------------------------------------*/
	for (int i = 0; i < k; ++i) { // For every iteration
		same_matrix = 1;
		cudaMemcpy(same_matrix_d, &same_matrix, sizeof(int), cudaMemcpyHostToDevice);
		calculateFrameShared <<< dimGrid, dimBlock >>> (G_d, GNext_d, w_d, n, same_matrix_d);

		cudaMemcpy(&same_matrix, same_matrix_d, sizeof(int), cudaMemcpyDeviceToHost); // Kernel to get flag indicating whether matrices are the same

		// Swap pointers
		int* ptri = G_d;
		G_d = GNext_d;
		GNext_d = ptri;

		// Exit if nothing changed
		if (same_matrix) {
			printf("Finished at %ith iteration. ", i);
			cudaFree(GNext_d);
			cudaFree(w_d);
			break;
		}
	}  // for every i < k

	// copy data from device and cleanup
	cudaMemcpy(G, G_d, n * n * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(G_d);
}
/////////////////////////////////////////////////