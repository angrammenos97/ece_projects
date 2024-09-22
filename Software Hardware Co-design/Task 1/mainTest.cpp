#include <iostream>
#include <cstdlib>
#include "mulArr.h"

void mulArrSW(uint8 A[n][m], uint8 B[m][p], uint32 C[n][p]);

int main(int argc, char *argv[])
{
	uint8 A[n][m], B[m][p];
	uint32 Ctest[n][p], Cactual[n][p];

	for(int i = 0; i < n; i++) {
		for(int j = 0; j < p; j++) {
			A[i][j] = (uint8)(rand() % 256);
			B[i][j] = (uint8)(rand() % 256);
			Ctest[i][j] = 0;
			Cactual[i][j] = 0;
		}
	}

	mulArrSW(A, B, Cactual);
	mulArrHW(A, B, Ctest);

	bool flag = false;
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < p; j++) {
			if(Cactual[i][j] != Ctest[i][j]) {
				flag = true;
				break;
			}
		}
		if(flag == true)
			break;
	}
	if(!flag)
		std::cout << "TEST PASSED with size = " << n << "x" << p << "\n";
	else
		std::cout << "TEST FAILED\n";

	return 0;
}

void mulArrSW(uint8 A[n][m], uint8 B[m][p], uint32 C[n][p])
{
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < p; j++) {
			for(int k = 0; k < m; k++) {
				C[i][j] += (uint32)(A[i][k]) * (uint32)(B[k][j]);
			}
		}
	}
}
