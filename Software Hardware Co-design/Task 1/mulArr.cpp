#include "mulArr.h"

//#define BRAM

void mulArrHW(uint8 A[n][m], uint8 B[m][p], uint32 C[n][p])
{
	const int N = n;
	const int M = m;
	const int P = p;

#ifdef BRAM
	uint8 BRAM_A[n][m], BRAM_B[m][p];
	#pragma HLS ARRAY_PARTITION variable=BRAM_A cyclic factor=M/2 dim=2
	#pragma HLS ARRAY_PARTITION variable=BRAM_B cyclic factor=M/2 dim=1

	for(int i = 0; i < n; i++) {
	#pragma HLS loop_tripcount min=N max=N
		for(int j = 0; j < m; j++) {
		#pragma HLS loop_tripcount min=M max=M
		#pragma HLS PIPELINE II=1
			BRAM_A[i][j] = A[i][j];
		}
	}
	for(int i = 0; i < m; i++) {
	#pragma HLS loop_tripcount min=M max=M
		for(int j = 0; j < p; j++) {
		#pragma HLS loop_tripcount min=P max=P
		#pragma HLS PIPELINE II=1
			BRAM_B[i][j] = B[i][j];
		}
	}
#else
	#pragma HLS ARRAY_PARTITION variable=A cyclic factor=M/2 dim=2
	#pragma HLS ARRAY_PARTITION variable=B cyclic factor=M/2 dim=1
#endif

	for(int i = 0; i < n; i++) {
	#pragma HLS loop_tripcount min=N max=N
		for(int j = 0; j < p; j++) {
		#pragma HLS loop_tripcount min=P max=P
		#pragma HLS PIPELINE II=1
			uint32 result = 0;
			for(int k = 0; k < m; k++) {
			#pragma HLS loop_tripcount min=M max=M
			#ifdef BRAM
				result += (uint32)(BRAM_A[i][k]) * (uint32)(BRAM_B[k][j]);
			#else
				result += (uint32)(A[i][k]) * (uint32)(B[k][j]);
			#endif
			}
			C[i][j] = result;
		}
	}
}
