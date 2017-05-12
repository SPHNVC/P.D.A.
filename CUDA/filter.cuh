#ifndef _filter_hpp
#define _filter_hpp

#include <algorithm>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include <iostream>
#include <vector>

/* Grid and Block definitions. Alter these as you please to tweak results. */
#define GRID_X 64
#define GRID_Y 64

#define MIN_RGB_VALUE 0
#define MAX_RGB_VALUE 255

typedef unsigned char uchar;
typedef unsigned int uint;

class Filter {
public:
	/**
	* Does a Sobel Filter on the input PGM image.
	* Return the copy-compute-copy time.
	*/
	double sobel_filter_gpu(const uchar * input, uchar * output, const uint height, const uint width);

	

	/* Flatten 2D array indices into 1D. */
	inline int get_array_index(const int x, const int y, const int width) {
		return y + x * width;
	}

	inline void start_timer() {
		this->timer = nullptr;
		sdkCreateTimer(&timer);
		sdkStartTimer(&timer);
	}

	inline double get_timer_value() {
		return sdkGetTimerValue(&timer);
	}

	/* Stops and deletes the timer object. */
	inline void stop_timer() {
		sdkStopTimer(&timer);
		sdkDeleteTimer(&timer);
	}

	inline std::string get_cuda_error() {
		return cudaGetErrorString(cudaGetLastError());
	}

	StopWatchInterface * timer;
};

#endif
