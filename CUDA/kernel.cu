#include "filter.cuh"


using namespace std;



__global__ void kernel_sobel_filter(const uchar * device_input_data, uchar * device_output_data, const uint height, const uint width) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	/* Bound check */
	if (x < 0 || x > width || y > height || y < 0)
		return;

	/* To detect horizontal lines. This is effectively the dx. */
	const int sobel_x[3][3] = {
		{ -1, 0, 1 },
		{ -2, 0, 2 },
		{ -1, 0, 1 }
	};
	/* To detect vertical lines. This is effectively the dy. */
	const int sobel_y[3][3] = {
		{ -1, -2, -1 },
		{ 0,   0,  0 },
		{ 1,   2,  1 }
	};

	double magnitude_x = 0;
	double magnitude_y = 0;

	for (uint j = 0; j < 3; ++j) {
		for (uint i = 0; i < 3; ++i) {
			const int x_focus = i + x;
			const int y_focus = j + y;
			const int index = y_focus + x_focus * width;
			magnitude_x += device_input_data[index] * sobel_x[i][j];
			magnitude_y += device_input_data[index] * sobel_y[i][j];
		}
	}
	double magnitude = sqrt(magnitude_x * magnitude_x + magnitude_y * magnitude_y);

	/* Edge cases of MIN or MAX RGB after the Sobel operator is applied */
	if (magnitude < MIN_RGB_VALUE)
		magnitude = MIN_RGB_VALUE;
	if (magnitude > MAX_RGB_VALUE)
		magnitude = MAX_RGB_VALUE;

	device_output_data[y + x * width] = magnitude;
}

/**
* Wrapper for calling the kernel.
*/
double Filter::sobel_filter_gpu(const uchar * host_data, uchar * output, const uint height, const uint width) {
	const int size = height * width * sizeof(uchar);

	/* Allocate device memory for the result. */
	/* Note that output to hold the HOST memory has already been allocated for. */
	void * device_input_data = nullptr;
	void * device_output_data = nullptr;

	if (cudaMalloc((void **)& device_input_data, size) != cudaSuccess)
		std::cerr << get_cuda_error() << std::endl;

	if (cudaMalloc((void **)& device_output_data, size) != cudaSuccess)
		std::cerr << get_cuda_error() << std::endl;

	/* Copy the input data to the device. */
	if (cudaMemcpy(device_input_data, host_data, size, cudaMemcpyHostToDevice) != cudaSuccess)
		std::cerr << get_cuda_error() << std::endl;

	/* Launch the kernel! */
	dim3 grid(GRID_X, GRID_Y, 1);
	dim3 block(EXPECTED_WIDTH / GRID_X, EXPECTED_HEIGHT / GRID_Y, 1);

	kernel_sobel_filter << <grid, block >> >((uchar*)device_input_data, (uchar*)device_output_data, height, width);

	if (cudaMemcpy(output, device_output_data, size, cudaMemcpyDeviceToHost) != cudaSuccess)
		std::cerr << get_cuda_error() << std::endl;

	cudaFree(device_input_data);
	cudaFree(device_output_data);

	/* Capture the device copy-compute-copy time. */
	return get_timer_value();
}

