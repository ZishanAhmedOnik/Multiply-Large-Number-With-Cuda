
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <cstring>
using namespace std;

__global__ void multiplyDigits(char* d_str1, char* d_str2, int* d_matrix) {
	int row = threadIdx.y;
	int col = threadIdx.x;

	int idx = row * blockDim.x + (col + (blockDim.y * row)) + 1 + (row);

	d_matrix[idx] = (d_str2[row] - '0') * (d_str1[col] - '0');
}

__global__ void propagateCarries(int* d_matrix, int numCols) {
	int idx = threadIdx.x * numCols;
	int carry = 0;

	for (int i = numCols - 1; i >= 0; i--) {
		int rowVal = (d_matrix[idx + i] + carry) % 10;
		carry = (d_matrix[idx + i] + carry) / 10;

		d_matrix[idx + i] = rowVal;
	}
}

__global__ void sumCols(int* d_matrix, int* d_result, int numRows, int numCols) {
	int sum = 0;
	
	for (int i = 0; i < numRows; i++) {
		sum += d_matrix[threadIdx.x + (numCols * i)];
	}

	d_result[threadIdx.x] = sum;
}

__host__ void propagateCarryInFinalResult(int* h_result, int numCols) {
	int carry = 0;

	for (int i = numCols - 1; i >= 0; i--) {
		int rowVal = (h_result[i] + carry) % 10;
		carry = (h_result[i] + carry) / 10;

		h_result[i] = rowVal;
	}
}

int main() {
	char* h_str1 = "123";
	char* h_str2 = "456";

	char* d_str1;
	char* d_str2;

	int* h_matrix;
	int* h_result;

	int* d_matrix;
	int* d_result;

	int row = strlen(h_str2);
	int col = strlen(h_str1) + row;

	h_matrix = new int[row * col];
	h_result = new int[col];

	cudaMalloc(&d_str1, sizeof(char) * strlen(h_str1));
	cudaMalloc(&d_str2, sizeof(char) * strlen(h_str2));

	cudaMalloc(&d_matrix, sizeof(int) * (row * col));
	cudaMemset(&d_matrix, 0, sizeof(int) * (row * col));

	cudaMalloc(&d_result, sizeof(int) * col);

	cudaMemcpy(d_str1, h_str1, sizeof(char) * strlen(h_str1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_str2, h_str2, sizeof(char) * strlen(h_str2), cudaMemcpyHostToDevice);

	multiplyDigits<<<1, dim3(3, 3)>>>(d_str1, d_str2, d_matrix);
	propagateCarries<<<1, 3>>>(d_matrix, col);
	sumCols<<<1, col>>>(d_matrix, d_result, row, col);

	cudaMemcpy(h_result, d_result, sizeof(int) * col, cudaMemcpyDeviceToHost);

	propagateCarryInFinalResult(h_result, col);

	for (int i = 0; i < col; i++) {
		cout << h_result[i];
	}
	cout << endl;

	cudaFree(d_str1);
	cudaFree(d_str2);
	cudaFree(d_result);

	delete[] h_matrix;
	delete[] h_result;

	cin.get();

	return 0;
}