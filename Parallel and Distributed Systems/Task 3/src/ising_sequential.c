//////////////// Ising Code Here ////////////////

#define WeightMatDim 5	// Weight Matrix Dimension
#define FloatError 1e-6

void calculateFrame(int* G, int* G_next, double* w, int n, int *same_matrix)
{
	for (int r_next = 0; r_next < n; r_next++) {	// for every row of the n*n space			
		for (int c_next = 0; c_next < n; c_next++) {	// for every point of the row of the n*n space
			double influence = 0.0;			// weighted influence of neighbors
			for (int x = -2; x <= 2; x++) {	// for every row of weight matrix
				int r = (r_next + x + n) % n;	// wrap around top with bottom
				for (int y = -2; y <= 2; y++) {	// for every weight of a row in weight matrix
					int c = (c_next + y + n) % n;	// wrap around left with right
					influence += *(G + r * n + c) * *(w + (x + 2) * WeightMatDim + (y + 2));	// +2 cause of the x and y offet
				}// for x < WeightMatDim
			}// for y < WeightMatDim

			// Update state
			if (influence > FloatError)			// apply threshold for floating point error
				*(G_next + r_next * n + c_next) = 1;
			else if (influence < -FloatError)	// apply threshold for floating point error
				*(G_next + r_next * n + c_next) = -1;
			else								// stay the same
				*(G_next + r_next * n + c_next) = *(G + r_next * n + c_next);

			// Did the magnetic moment changed?
			if (*(G_next + r_next * n + c_next) != *(G + r_next * n + c_next))
				*same_matrix = 0;

		}// for c_next < n
	}// for r_next < n
}

void ising(int *G, double *w, int k, int n)
{
	// Initialize memory
	int *G_next = (int*)malloc(n*n * sizeof(int));	// second array to swap pointers
	memcpy(G_next, G, n*n * sizeof(int));
	int same_matrix = 1;	// device parameter to hold if iterations should proceed

	int i = 0;
	for (i = 0; i < k; i++) {	// for every k iteration
		*same_matrix = 1;
		calculateFrame(G, G_next, w, n, &same_matrix);

		// Swap pointers
		int *ptri = G;
		G = G_next;
		G_next = ptri;

		// Exit if nothing changed
		if (same_matrix) {
			printf("Finished at %ith iteration. ", i);
			break;
		}
	}// for i < k

	//	Check which pointer is being used and free the other to save memory
	if ((i % 2) == 1) {
		memcpy(G_next, G, n * n * sizeof(int));
		free(G);
		return;
	}

	free(G_next);
}
/////////////////////////////////////////////////