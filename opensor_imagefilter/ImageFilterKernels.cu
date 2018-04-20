#include "ImageFilter.h"

#define BLOCKWIDTH 32
#define BLOCKHEIGHT 12
#define STRIDEALIGNMENT 32

//////////////////////////////////////////////////////////////////////////////
/// MAXIMUM/MINIMUM DETECTION
/// Given edge image as input and float3 (level of curvature) as output
//////////////////////////////////////////////////////////////////////////////
__global__ void MaxMinDetectionKernel(uchar3* input, uchar3* output, int width, int height, int stride) {
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;
	const int pos = ix + iy * stride;
	if (ix >= width || iy >= height) return;

	uchar3 pix = input[pos]; //center value
	uchar gray = (uchar)(((int)pix.x + (int)pix.y + (int)pix.z) / 3);

	int kSize = 5;
	int thresh = 10;
	int kOffset = (kSize - 1) / 2;
	bool isMax = false;
	bool isMin = false;
	int maxCount = 0;
	int minCount = 0;
	int totalCount = 0;
	//detect min or max per channel
	/*for (int j = -kOffset; j <= kOffset; j++) {
	for (int i = -kOffset; i <= kOffset; i++) {
	int col = (ix + i);
	int row = (iy + j);
	if ((col >= 0) && (col < width) && (row >= 0) && (row < height) && (pos != col + stride*row)) {
	totalCount++;
	uchar3 pixN = input[col + stride*row];
	if ((pix.x - pixN.x > thresh) && (pix.y - pixN.y > thresh) && (pix.z - pixN.z > thresh)) {
	maxCount++;
	}
	if ((pixN.x - pix.x > thresh) && (pixN.y - pix.y > thresh) && (pixN.z - pix.z > thresh)) {
	minCount++;
	}
	}
	}
	}*/
	for (int j = -kOffset; j <= kOffset; j++) {
		for (int i = -kOffset; i <= kOffset; i++) {
			int col = (ix + i);
			int row = (iy + j);
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height) && (pos != col + stride * row)) {
				totalCount++;
				uchar3 pixN = input[col + stride * row];
				uchar grayN = (uchar)(((int)pixN.x + (int)pixN.y + (int)pixN.z) / 3);
				if (gray - grayN > thresh) {
					maxCount++;
				}
				if (grayN - gray > thresh) {
					minCount++;
				}
			}
		}
	}
	if (maxCount == totalCount) {
		output[pos] = make_uchar3(0, 0, 255);
	}
	else if (minCount == totalCount) {
		output[pos] = make_uchar3(0, 255, 0);
	}
	else {
		output[pos] = make_uchar3(pix.x / 2, pix.y / 2, pix.z / 2);
	}
}

void ImageFilter::MinMaxDetection(uchar3* input, uchar3* output, int w, int h, int s) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	MaxMinDetectionKernel << < blocks, threads >> > (input, output, w, h, s);
}
//////////////////////////////////////////////////////////////////////////////
/// CURVE DETECTION
/// Given edge image as input and float3 (level of curvature) as output
//////////////////////////////////////////////////////////////////////////////
__device__ int findNeighbor(uchar img[], uchar label[], int index)
{

}

__global__ void CurveDetectionKernel(uchar *input, float3 *output, int width, int height, int stride) {
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;
	const int pos = ix + iy * stride;
	if (ix >= width || iy >= height) return;

	uchar pix = input[pos];
	int xtotal = 0;
	int ytotal = 0;
	int total = 0;
	bool isLine = false;
	int kSize = 5;
	int kOffset = (kSize - 1) / 2;

	uchar label[25] = { 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0
	};

	uchar img[25] = { 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0
	};

	if (pix == 255) {
		//copy data to img
		for (int j = -kOffset; j <= kOffset; j++) {
			for (int i = -kOffset; i <= kOffset; i++) {
				int col = (ix + i);
				int row = (iy + j);
				if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
					img[(j + kOffset) * 5 + (i + kOffset)] = input[col + stride * row];
				}
			}
		}

		//iterate through img to detect the line and save to label
		//int isLine = 0;
		int lastLayerCount = 0;
		for (int iter = 0; iter < 3; iter++) {
			for (int j = 0; j < kSize; j++) {
				for (int i = 0; i < kSize; i++) {
					if (label[j * kSize + i] == 1) {
						if (iter == 2) {//last layer, wrong if there are two neighboring outerlayer pixels
							lastLayerCount++;
							label[j * kSize + i] == 2;
						}
						else {//check 8-neighbor
							int count = 0;
							for (int m = -1; m < 2; m++) {
								for (int n = -1; n < 2; n++) {
									if ((img[(j + m) * 5 + (i + n)] == 255) && (label[(j + m) * 5 + (i + n)] == 0)) {
										count++;
										label[(j + m) * 5 + (i + n)] = 1;
									}
								}
							}
							if (count > 0) {
								label[j * kSize + i] = 2;
							}
						}
					}
				}
			}
		}
		if (lastLayerCount >= 2) {
			isLine = true;
		}

		//detect curvature
		//	for (int j = -2; j < 3; j++) {
		//		for (int i = -2; i < 3; i++) {
		//			//get values
		//			int col = (ix + i);
		//			int row = (iy + j);
		//			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
		//				if (input[col + stride*row] == 255) {
		//					xtotal += i;
		//					ytotal += j;
		//					total++;
		//				}
		//			}
		//		}
		//	}
		//	float xval = (float)xtotal / (float)total;
		//	float yval = (float)ytotal / (float)total;
		//	float mag = sqrt(xval*xval + yval*yval);
		//	if (mag > 0.5) {
		//		output[pos] = make_float3(0, 0, 1.0);
		//	}
		//	else {
		//		output[pos] = make_float3(1, 1, 1);
		//	}
		//}
		//else {
		//	output[pos] = make_float3(0, 0, 0);
	}
	if (isLine) {
		output[pos] = make_float3(1, 1, 1);
		//output[pos] = make_float3(img[12], img[12], img[12]);
	}
	else {
		output[pos] = make_float3(0, 0, 0);
	}
	//output[pos] = make_float3((float)input[pos], (float)input[pos], (float)input[pos]);
}

void ImageFilter::CurveDetection(uchar* input, float3 *output, int w, int h, int s) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	CurveDetectionKernel << < blocks, threads >> > (input, output, w, h, s);
}


//////////////////////////////////////////////////////////////////////////////
///RGB to HSV
//////////////////////////////////////////////////////////////////////////////
__device__ float3 convert_one_pixel_to_hsv(uchar3 pixel) {
	float r, g, b;
	float h, s, v;

	r = pixel.x / 255.0f;
	g = pixel.y / 255.0f;
	b = pixel.z / 255.0f;

	float max = fmax(r, fmax(g, b));
	float min = fmin(r, fmin(g, b));
	float diff = max - min;

	v = max;

	if (v == 0.0f) { // black
		h = s = 0.0f;
	}
	else {
		s = diff / v;
		if (diff < 0.001f) { // grey
			h = 0.0f;
		}
		else { // color
			if (max == r) {
				h = 60.0f * (g - b) / diff;
				if (h < 0.0f) { h += 360.0f; }
			}
			else if (max == g) {
				h = 60.0f * (2 + (b - r) / diff);
			}
			else {
				h = 60.0f * (4 + (r - g) / diff);
			}
		}
	}

	return make_float3(h, s, v);
}

__global__ void RgbToHsvKernel(uchar3 *rgb, float3 *hsv, int width, int height, int stride) {
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;
	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;
	uchar3 rgb_pixel = rgb[pos];
	float3 hsv_pixel = convert_one_pixel_to_hsv(rgb_pixel);
	hsv[pos] = hsv_pixel;
}

void ImageFilter::RgbToHsv(uchar3* input, float3* output, int w, int h, int s) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	RgbToHsvKernel << < blocks, threads >> > (input, output, w, h, s);
}

//////////////////////////////////////////////////////////////////////////////
/// Uniform Filter
//////////////////////////////////////////////////////////////////////////////
__global__
void UniformFilterKernel(uchar3* input, uchar3* output, int width, int height, int stride, int kernelsize)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;
	const int pos = ix + iy * stride;

	extern __shared__ uchar3 sinput[];

	if (ix >= width || iy >= height) return;

	//fill shared memory
	int offset = (kernelsize - 1) / 2;
	const int six = threadIdx.x + offset;
	const int siy = threadIdx.y + offset;
	int sharedStride = blockDim.x + kernelsize - 1;
	sinput[siy * sharedStride + six] = input[pos];
	//pad for edges and corners
	if (threadIdx.x == 0) { //left edge
		for (int k = -offset; k < 0; k++) {
			if ((ix + k) < 0)
				sinput[siy * sharedStride + six + k] = input[ix + iy * stride];
			else
				sinput[siy * sharedStride + six + k] = input[(ix + k) + iy * stride];
		}
	}
	else if (threadIdx.x == (blockDim.x - 1)) { //right edge
		for (int k = 1; k <= offset; k++) {
			if ((ix + k) > width - 1)
				sinput[siy * sharedStride + six + k] = input[ix + iy * stride];
			else
				sinput[siy * sharedStride + six + k] = input[(ix + k) + iy * stride];
		}
	}
	if (threadIdx.y == 0) { //top edge
		for (int k = -offset; k < 0; k++) {
			if ((iy + k) < 0)
				sinput[six + (siy + k)*sharedStride] = input[ix + iy * stride];
			else
				sinput[six + (siy + k)*sharedStride] = input[ix + (iy + k)*stride];
		}
		if (threadIdx.x == 0) { //top left corner
			for (int ky = -offset; ky < 0; ky++) {
				for (int kx = -offset; kx < 0; kx++) {
					if ((ix + kx) < 0 && (iy + ky) < 0)
						sinput[(siy + ky) * sharedStride + (six + kx)] = input[ix + iy * stride];
					else if ((ix + kx) < 0)
						sinput[(siy + ky) * sharedStride + (six + kx)] = input[ix + (iy + ky)*stride];
					else if ((iy + ky) < 0)
						sinput[(siy + ky) * sharedStride + (six + kx)] = input[(ix + kx) + iy * stride];
					else
						sinput[(siy + ky) * sharedStride + (six + kx)] = input[(ix + kx) + (iy + ky)*stride];
				}
			}
		}
		else if (threadIdx.x == (blockDim.x - 1)) { //top right corner
			for (int ky = -offset; ky < 0; ky++) {
				for (int kx = 1; kx <= offset; kx++) {
					if ((ix + kx) > width - 1 && (iy + ky) < 0)
						sinput[(siy + ky) * sharedStride + (six + kx)] = input[ix + iy * stride];
					else if ((ix + kx) > width - 1)
						sinput[(siy + ky) * sharedStride + (six + kx)] = input[ix + (iy + ky)*stride];
					else if ((iy + ky) < 0)
						sinput[(siy + ky) * sharedStride + (six + kx)] = input[(ix + kx) + iy * stride];
					else
						sinput[(siy + ky) * sharedStride + (six + kx)] = input[(ix + kx) + (iy + ky)*stride];
				}
			}
		}
	}
	else if (threadIdx.y == (blockDim.y - 1)) { //bottom edge
		for (int k = 1; k <= offset; k++) {
			if ((iy + k) > height - 1)
				sinput[six + (siy + k)*sharedStride] = input[ix + iy * stride];
			else
				sinput[six + (siy + k)*sharedStride] = input[ix + (iy + k)*stride];
		}
		if (threadIdx.x == 0) { //bottom left corner
			for (int ky = 1; ky <= offset; ky++) {
				for (int kx = -offset; kx < 0; kx++) {
					if ((ix + kx) < 0 && (iy + ky) > height - 1)
						sinput[(siy + ky) * sharedStride + (six + kx)] = input[ix + iy * stride];
					else if ((ix + kx) < 0)
						sinput[(siy + ky) * sharedStride + (six + kx)] = input[ix + (iy + ky)*stride];
					else if ((iy + ky) > height - 1)
						sinput[(siy + ky) * sharedStride + (six + kx)] = input[(ix + kx) + iy * stride];
					else
						sinput[(siy + ky) * sharedStride + (six + kx)] = input[(ix + kx) + (iy + ky)*stride];
				}
			}
		}
		else if (threadIdx.x == (blockDim.x - 1)) { //bottom right corner
			for (int ky = 1; ky <= offset; ky++) {
				for (int kx = 1; kx <= offset; kx++) {
					if ((ix + kx) > width - 1 && (iy + ky) > height - 1)
						sinput[(siy + ky) * sharedStride + (six + kx)] = input[ix + iy * stride];
					else if ((ix + kx) > width - 1)
						sinput[(siy + ky) * sharedStride + (six + kx)] = input[ix + (iy + ky)*stride];
					else if ((iy + ky) > height - 1)
						sinput[(siy + ky) * sharedStride + (six + kx)] = input[(ix + kx) + iy * stride];
					else
						sinput[(siy + ky) * sharedStride + (six + kx)] = input[(ix + kx) + (iy + ky)*stride];
				}
			}
		}
	}
	__syncthreads();

	int3 sum = make_int3(0, 0, 0);
	int total = 0;
	//int shift = (kernelsize - 1) / 2;

	for (int j = 0; j < kernelsize; j++) {
		for (int i = 0; i < kernelsize; i++) {
			int col = (six + i - offset);
			int row = (siy + j - offset);
			//if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
			sum.x += (int)sinput[col + sharedStride * row].x;
			sum.y += (int)sinput[col + sharedStride * row].y;
			sum.z += (int)sinput[col + sharedStride * row].z;
			total++;
			//}
		}
	}
	if (total > 0) {
		output[pos].x = (uchar)((float)sum.x / (float)total);
		output[pos].y = (uchar)((float)sum.y / (float)total);
		output[pos].z = (uchar)((float)sum.z / (float)total);
	}
	else {
		output[pos] = make_uchar3(0, 0, 0);
	}
	//output[pos] = sinput[(six - offset) + (siy - offset)*sharedStride];
}

__global__
void UniformFilterKernel(uchar* input, uchar* output, int width, int height, int stride, int kernelsize)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;
	const int pos = ix + iy * stride;
	if (ix >= width || iy >= height) return;
	int sum = 0.0f;
	int total = 0;
	int shift = (kernelsize - 1) / 2;
	for (int j = 0; j < kernelsize; j++) {
		for (int i = 0; i < kernelsize; i++) {
			int col = (ix + i - shift);
			int row = (iy + j - shift);
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				sum += (int)input[col + stride * row];
				total++;
			}
		}
	}
	if (total > 0) {
		output[pos] = (uchar)((float)sum / (float)total);
	}
	else output[pos] = (uchar)0;
}

__global__
void UniformFilterKernel_global(uchar3* input, uchar3* output, int width, int height, int stride, int kernelsize)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;
	const int pos = ix + iy * stride;
	if (ix >= width || iy >= height) return;
	int3 sum = make_int3(0, 0, 0);
	int total = 0;
	int shift = (kernelsize - 1) / 2;
	for (int j = 0; j < kernelsize; j++) {
		for (int i = 0; i < kernelsize; i++) {
			int col = (ix + i - shift);
			int row = (iy + j - shift);
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				sum.x += (int)input[col + stride * row].x;
				sum.y += (int)input[col + stride * row].y;
				sum.z += (int)input[col + stride * row].z;
				total++;
			}
		}
	}
	if (total > 0) {
		output[pos].x = (uchar)((float)sum.x / (float)total);
		output[pos].y = (uchar)((float)sum.y / (float)total);
		output[pos].z = (uchar)((float)sum.z / (float)total);
	}
	else {
		output[pos] = make_uchar3(0, 0, 0);
	}
	//output[pos] = input[pos];
}

void ImageFilter::UniformFilter(uchar3* input, uchar3* output, int w, int h, int s, int kernelsize) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	//UniformFilterKernel << < blocks, threads >> > (input, output, w, h, s, kernelsize); //global memory
	int sharedMemSize = (BlockWidth + (kernelsize - 1))*(BlockHeight + (kernelsize - 1))*(sizeof(uchar3));
	UniformFilterKernel << < blocks, threads, sharedMemSize >> > (input, output, w, h, s, kernelsize); //shared memory
																									   //UniformFilterKernel_global << < blocks, threads>> > (input, output, w, h, s, kernelsize); //shared memory
}


//////////////////////////////////////////////////////////////////////////////
/// Median Filter
//////////////////////////////////////////////////////////////////////////////
__global__
void MedianFilterKernel(uchar3* input, uchar3* output, int width, int height, int stride, int kernelsize)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;
	const int pos = ix + iy * stride;
	if (ix >= width || iy >= height) return;

	float mu[25] = { 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0 };

	uchar R[25] = { 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0 };

	uchar G[25] = { 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0 };

	uchar B[25] = { 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0 };

	//convert to gray
	uchar3 val;
	int total = 0;
	float mean;
	for (int j = 0; j < 5; j++) {
		for (int i = 0; i < 5; i++) {
			//get values
			int col = (ix + i - 2);
			int row = (iy + j - 2);
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				val = input[col + stride * row];
			}
			else if ((col < 0) && (row >= 0) && (row < height)) {
				val = input[stride*row];
			}
			else if ((col >= width) && (row >= 0) && (row < height)) {
				val = input[width - 1 + stride * row];
			}
			else if ((col >= 0) && (col < width) && (row < 0)) {
				val = input[col];
			}
			else if ((col >= 0) && (col < width) && (row >= height)) {
				val = input[col + stride * (height - 1)];
			}
			R[j * 5 + i] = val.x;
			G[j * 5 + i] = val.y;
			B[j * 5 + i] = val.z;
			mu[j * 5 + i] = ((float)val.x + (float)val.y + (float)val.z) / 3.0f;
			total += mu[j * 5 + i];
		}
	}
	mean = (float)total / 25.0f;

	float tmpu, tmpv;
	int index[25] = { 0, 1, 2, 3, 4, 5,
		6, 7, 8, 9, 10,
		11, 12, 13, 14, 15,
		16, 17, 18, 19, 20,
		21, 22, 23, 24 };

	//Brute Force
	//for (int j = 0; j < 25; j++) {
	//	for (int i = j + 1; i < 25; i++) {
	//		if (mu[j] > mu[i]) {
	//			//Swap the variables.
	//			tmpu = mu[j];
	//			mu[j] = mu[i];
	//			mu[i] = tmpu;
	//			tmpv = index[j];
	//			index[j] = index[i];
	//			index[i] = tmpv;
	//		}
	//	}
	//}
	//output[pos] = make_uchar3(R[index[12]], G[index[12]], B[index[12]]);

	//False Median
	/*float closest = mu[0];
	float diff = abs(mean - mu[0]);
	int closest_index = 0;
	for (int j = 1; j < 25; j++) {
	if (abs(mean - mu[j]) < diff) {
	closest = mu[j];
	diff = abs(mean - mu[j]);
	closest_index = j;
	}
	}
	output[pos] = make_uchar3(R[closest_index], G[closest_index], B[closest_index]);*/

	//Lowest 13
	for (int j = 0; j < 13; j++) {
		for (int i = j + 1; i < 25; i++) {
			if (mu[j] > mu[i]) {
				//Swap the variables.
				tmpu = mu[j];
				mu[j] = mu[i];
				mu[i] = tmpu;
				tmpv = index[j];
				index[j] = index[i];
				index[i] = tmpv;
			}
		}
	}
	output[pos] = make_uchar3(R[index[12]], G[index[12]], B[index[12]]);
	//output[pos] = make_uchar3(R[12], G[12], B[12]);
}

__global__
void MedianFilterKernel3(uchar3* input, uchar3* output, int width, int height, int stride, int kernelsize)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;
	const int pos = ix + iy * stride;
	if (ix >= width || iy >= height) return;

	float mu[9] = { 0, 0, 0, 0, 0,
		0, 0, 0, 0 };

	uchar R[9] = { 0, 0, 0, 0, 0,
		0, 0, 0, 0 };

	uchar G[9] = { 0, 0, 0, 0, 0,
		0, 0, 0, 0 };

	uchar B[9] = { 0, 0, 0, 0, 0,
		0, 0, 0, 0 };

	//convert to gray
	uchar3 val;
	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < 3; i++) {
			//get values
			int col = (ix + i - 1);
			int row = (iy + j - 1);
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				val = input[col + stride * row];
			}
			else if ((col < 0) && (row >= 0) && (row < height)) {
				val = input[stride*row];
			}
			else if ((col >= width) && (row >= 0) && (row < height)) {
				val = input[width - 1 + stride * row];
			}
			else if ((col >= 0) && (col < width) && (row < 0)) {
				val = input[col];
			}
			else if ((col >= 0) && (col < width) && (row >= height)) {
				val = input[col + stride * (height - 1)];
			}
			R[j * 3 + i] = val.x;
			G[j * 3 + i] = val.y;
			B[j * 3 + i] = val.z;
			mu[j * 3 + i] = ((float)val.x + (float)val.y + (float)val.z) / 3.0f;
		}
	}

	float tmpu, tmpv;
	int index[9] = { 0, 1, 2, 3, 4, 5,
		6, 7, 8 };

	for (int j = 0; j < 9; j++) {
		for (int i = j + 1; i < 9; i++) {
			if (mu[j] > mu[i]) {
				//Swap the variables.
				tmpu = mu[j];
				mu[j] = mu[i];
				mu[i] = tmpu;
				tmpv = index[j];
				index[j] = index[i];
				index[i] = tmpv;
			}
		}
	}
	output[pos] = make_uchar3(R[index[5]], G[index[5]], B[index[5]]);
}

void ImageFilter::MedianFilter(uchar3* input, uchar3* output, int w, int h, int s, int kernelsize) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	if (kernelsize == 3) {
		MedianFilterKernel3 << < blocks, threads >> > (input, output, w, h, s, kernelsize);
	}
	else {
		MedianFilterKernel << < blocks, threads >> > (input, output, w, h, s, kernelsize);
	}
}


/////////////////////////////////////////////////////////////////////////////
/// Edge Detection
/////////////////////////////////////////////////////////////////////////////
__global__
void EdgeDetectKernel(uchar3* input, float* output, int width, int height, int stride, int kernelsize, float threshold) {
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;
	const int pos = ix + iy * stride;
	if (ix >= width || iy >= height) return;
	float3 sum = make_float3(0.0f, 0.0f, 0.0f);
	float filter[9] = { -1.0f, -2.0f, -1.0f, -2.0f, 12.0f, -2.0f, -1.0f, -2.0f, -1.0f };
	//float filter[9] = { -2.0f, -2.0f, -2.0f, -2.0f, 16.0f, -2.0f, -2.0f, -2.0f, -2.0f };
	float total = 0.0f;
	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < 3; i++) {
			int col = (ix + i - 1);
			int row = (iy + j - 1);
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				sum.x += filter[j * 3 + i] * (float)input[col + stride * row].x / 256.0f;
				sum.y += filter[j * 3 + i] * (float)input[col + stride * row].y / 256.0f;
				sum.z += filter[j * 3 + i] * (float)input[col + stride * row].z / 256.0f;
			}
			else if ((col < 0) && (row >= 0) && (row < height)) {
				sum.x += filter[j * 3 + i] * (float)input[stride*row].x / 256.0f;
				sum.y += filter[j * 3 + i] * (float)input[stride*row].y / 256.0f;
				sum.z += filter[j * 3 + i] * (float)input[stride*row].z / 256.0f;
			}
			else if ((col >= width) && (row >= 0) && (row < height)) {
				sum.x += filter[j * 3 + i] * (float)input[width - 1 + stride * row].x / 256.0f;
				sum.y += filter[j * 3 + i] * (float)input[width - 1 + stride * row].y / 256.0f;
				sum.z += filter[j * 3 + i] * (float)input[width - 1 + stride * row].z / 256.0f;
			}
			else if ((col >= 0) && (col < width) && (row < 0)) {
				sum.x += filter[j * 3 + i] * (float)input[col].x / 256.0f;
				sum.y += filter[j * 3 + i] * (float)input[col].y / 256.0f;
				sum.z += filter[j * 3 + i] * (float)input[col].z / 256.0f;
			}
			else if ((col >= 0) && (col < width) && (row >= height)) {
				sum.x += filter[j * 3 + i] * (float)input[col + stride * (height - 1)].x / 256.0f;
				sum.y += filter[j * 3 + i] * (float)input[col + stride * (height - 1)].y / 256.0f;
				sum.z += filter[j * 3 + i] * (float)input[col + stride * (height - 1)].z / 256.0f;
			}
		}
	}
	float edge_r = abs(sum.x);
	float edge_g = abs(sum.y);
	float edge_b = abs(sum.z);
	//thresholding
	if ((edge_r > threshold) || (edge_g > threshold) || (edge_b > threshold)) output[pos] = 1.0f;
	else output[pos] = 0.0f;
	//output[pos] = edge_r;
}

__global__
void EdgeDetectKernel(float3* input, float* output, int width, int height, int stride, int kernelsize, float threshold) {
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;
	const int pos = ix + iy * stride;
	if (ix >= width || iy >= height) return;
	float3 sum = make_float3(0.0f, 0.0f, 0.0f);
	float filter[9] = { -1.0f, -2.0f, -1.0f, -2.0f, 12.0f, -2.0f, -1.0f, -2.0f, -1.0f };
	//float filter[9] = { -2.0f, -2.0f, -2.0f, -2.0f, 16.0f, -2.0f, -2.0f, -2.0f, -2.0f };
	float total = 0.0f;
	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < 3; i++) {
			int col = (ix + i - 1);
			int row = (iy + j - 1);
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				sum.x += filter[j * 3 + i] * (float)input[col + stride * row].x / 256.0f;
				sum.y += filter[j * 3 + i] * (float)input[col + stride * row].y / 256.0f;
				sum.z += filter[j * 3 + i] * (float)input[col + stride * row].z / 256.0f;
			}
			else if ((col < 0) && (row >= 0) && (row < height)) {
				sum.x += filter[j * 3 + i] * (float)input[stride*row].x / 256.0f;
				sum.y += filter[j * 3 + i] * (float)input[stride*row].y / 256.0f;
				sum.z += filter[j * 3 + i] * (float)input[stride*row].z / 256.0f;
			}
			else if ((col >= width) && (row >= 0) && (row < height)) {
				sum.x += filter[j * 3 + i] * (float)input[width - 1 + stride * row].x / 256.0f;
				sum.y += filter[j * 3 + i] * (float)input[width - 1 + stride * row].y / 256.0f;
				sum.z += filter[j * 3 + i] * (float)input[width - 1 + stride * row].z / 256.0f;
			}
			else if ((col >= 0) && (col < width) && (row < 0)) {
				sum.x += filter[j * 3 + i] * (float)input[col].x / 256.0f;
				sum.y += filter[j * 3 + i] * (float)input[col].y / 256.0f;
				sum.z += filter[j * 3 + i] * (float)input[col].z / 256.0f;
			}
			else if ((col >= 0) && (col < width) && (row >= height)) {
				sum.x += filter[j * 3 + i] * (float)input[col + stride * (height - 1)].x / 256.0f;
				sum.y += filter[j * 3 + i] * (float)input[col + stride * (height - 1)].y / 256.0f;
				sum.z += filter[j * 3 + i] * (float)input[col + stride * (height - 1)].z / 256.0f;
			}
		}
	}
	float edge_r = abs(sum.x);
	float edge_g = abs(sum.y);
	float edge_b = abs(sum.z);
	//thresholding
	if ((edge_r > threshold) || (edge_g > threshold) || (edge_b > threshold)) output[pos] = 1.0f;
	else output[pos] = 0.0f;
	//output[pos] = edge_r;
}

__global__
void EdgeDetectHsvKernel(float3* input, float* output, int width, int height, int stride, int kernelsize, float3 threshold) {
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;
	const int pos = ix + iy * stride;
	if (ix >= width || iy >= height) return;
	float3 sum = make_float3(0.0f, 0.0f, 0.0f);
	float filter[9] = { -1.0f, -2.0f, -1.0f, -2.0f, 12.0f, -2.0f, -1.0f, -2.0f, -1.0f };
	//float filter[9] = { -2.0f, -2.0f, -2.0f, -2.0f, 16.0f, -2.0f, -2.0f, -2.0f, -2.0f };
	float total = 0.0f;
	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < 3; i++) {
			int col = (ix + i - 1);
			int row = (iy + j - 1);
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				sum.x += filter[j * 3 + i] * (float)input[col + stride * row].x / 256.0f;
				sum.y += filter[j * 3 + i] * (float)input[col + stride * row].y / 256.0f;
				sum.z += filter[j * 3 + i] * (float)input[col + stride * row].z / 256.0f;
			}
			else if ((col < 0) && (row >= 0) && (row < height)) {
				sum.x += filter[j * 3 + i] * (float)input[stride*row].x / 256.0f;
				sum.y += filter[j * 3 + i] * (float)input[stride*row].y / 256.0f;
				sum.z += filter[j * 3 + i] * (float)input[stride*row].z / 256.0f;
			}
			else if ((col >= width) && (row >= 0) && (row < height)) {
				sum.x += filter[j * 3 + i] * (float)input[width - 1 + stride * row].x / 256.0f;
				sum.y += filter[j * 3 + i] * (float)input[width - 1 + stride * row].y / 256.0f;
				sum.z += filter[j * 3 + i] * (float)input[width - 1 + stride * row].z / 256.0f;
			}
			else if ((col >= 0) && (col < width) && (row < 0)) {
				sum.x += filter[j * 3 + i] * (float)input[col].x / 256.0f;
				sum.y += filter[j * 3 + i] * (float)input[col].y / 256.0f;
				sum.z += filter[j * 3 + i] * (float)input[col].z / 256.0f;
			}
			else if ((col >= 0) && (col < width) && (row >= height)) {
				sum.x += filter[j * 3 + i] * (float)input[col + stride * (height - 1)].x / 256.0f;
				sum.y += filter[j * 3 + i] * (float)input[col + stride * (height - 1)].y / 256.0f;
				sum.z += filter[j * 3 + i] * (float)input[col + stride * (height - 1)].z / 256.0f;
			}
		}
	}
	float edge_h = abs(sum.x);
	float edge_s = abs(sum.y);
	float edge_v = abs(sum.z);
	//thresholding
	if ((edge_h > threshold.x) || (edge_s > threshold.y) || (edge_v > threshold.z)) output[pos] = 1.0f;
	//if (edge_s > threshold) output[pos] = 1.0f;
	else output[pos] = 0.0f;
	output[pos] = edge_h;
}

void ImageFilter::EdgeDetect(uchar3* input, float* output, int w, int h, int s, int kernelsize, float threshold) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	EdgeDetectKernel << < blocks, threads >> > (input, output, w, h, s, kernelsize, threshold);
}

void ImageFilter::EdgeDetect(float3* input, float* output, int w, int h, int s, int kernelsize, float threshold) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	EdgeDetectKernel << < blocks, threads >> > (input, output, w, h, s, kernelsize, threshold);
}

void ImageFilter::EdgeDetectHsv(float3* input, float* output, int w, int h, int s, int kernelsize, float3 threshold) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	EdgeDetectHsvKernel << < blocks, threads >> > (input, output, w, h, s, kernelsize, threshold);
}


/////////////////////////////////////////////////////////////////////////////
/// Gaussian Blur
/////////////////////////////////////////////////////////////////////////////
__global__
void GaussianBlurKernel(uchar3* input, uchar3* output, int width, int height, int stride, int kernelsize) {
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;
	const int pos = ix + iy * stride;
	if (ix >= width || iy >= height) return;
	float3 sum = make_float3(0.0f, 0.0f, 0.0f);
	float filter[9] = { 0.0625f, 0.125f, 0.0625f,
		0.125f, 0.25f, 0.125f,
		0.0625f, 0.125f, 0.0625f };
	float total = 0.0f;
	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < 3; i++) {
			int col = (ix + i - 1);
			int row = (iy + j - 1);
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				sum.x += filter[j * 3 + i] * (float)input[col + stride * row].x;
				sum.y += filter[j * 3 + i] * (float)input[col + stride * row].y;
				sum.z += filter[j * 3 + i] * (float)input[col + stride * row].z;
			}
			else if ((col < 0) && (row >= 0) && (row < height)) {
				sum.x += filter[j * 3 + i] * (float)input[stride*row].x;
				sum.y += filter[j * 3 + i] * (float)input[stride*row].y;
				sum.z += filter[j * 3 + i] * (float)input[stride*row].z;
			}
			else if ((col >= width) && (row >= 0) && (row < height)) {
				sum.x += filter[j * 3 + i] * (float)input[width - 1 + stride * row].x;
				sum.y += filter[j * 3 + i] * (float)input[width - 1 + stride * row].y;
				sum.z += filter[j * 3 + i] * (float)input[width - 1 + stride * row].z;
			}
			else if ((col >= 0) && (col < width) && (row < 0)) {
				sum.x += filter[j * 3 + i] * (float)input[col].x;
				sum.y += filter[j * 3 + i] * (float)input[col].y;
				sum.z += filter[j * 3 + i] * (float)input[col].z;
			}
			else if ((col >= 0) && (col < width) && (row >= height)) {
				sum.x += filter[j * 3 + i] * (float)input[col + stride * (height - 1)].x;
				sum.y += filter[j * 3 + i] * (float)input[col + stride * (height - 1)].y;
				sum.z += filter[j * 3 + i] * (float)input[col + stride * (height - 1)].z;
			}
		}
	}
	output[pos].x = (uchar)sum.x;
	output[pos].y = (uchar)sum.y;
	output[pos].z = (uchar)sum.z;
}

__global__
void GaussianBlur5Kernel(uchar3* input, uchar3* output, int width, int height, int stride, int kernelsize) {
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;
	const int pos = ix + iy * stride;
	if (ix >= width || iy >= height) return;
	float3 sum = make_float3(0.0f, 0.0f, 0.0f);
	float filter[25] = { 0.00390625f, 0.015625f, 0.0234375f, 0.015625f, 0.00390625f,
		0.015625f, 0.0625f, 0.09375f, 0.0625f,0.015625f,
		0.0234375f, 0.09375f, 0.140625f, 0.09375f, 0.0234375f,
		0.015625f, 0.0625f, 0.09375f, 0.0625f, 0.015625,
		0.00390625f, 0.015625f, 0.0234375f, 0.015625f,0.00390625f };
	float total = 0.0f;
	for (int j = 0; j < 5; j++) {
		for (int i = 0; i < 5; i++) {
			int col = (ix + i - 2);
			int row = (iy + j - 2);
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				sum.x += filter[j * 5 + i] * (float)input[col + stride * row].x;
				sum.y += filter[j * 5 + i] * (float)input[col + stride * row].y;
				sum.z += filter[j * 5 + i] * (float)input[col + stride * row].z;
			}
			else if ((col < 0) && (row >= 0) && (row < height)) {
				sum.x += filter[j * 5 + i] * (float)input[stride*row].x;
				sum.y += filter[j * 5 + i] * (float)input[stride*row].y;
				sum.z += filter[j * 5 + i] * (float)input[stride*row].z;
			}
			else if ((col >= width) && (row >= 0) && (row < height)) {
				sum.x += filter[j * 5 + i] * (float)input[width - 1 + stride * row].x;
				sum.y += filter[j * 5 + i] * (float)input[width - 1 + stride * row].y;
				sum.z += filter[j * 5 + i] * (float)input[width - 1 + stride * row].z;
			}
			else if ((col >= 0) && (col < width) && (row < 0)) {
				sum.x += filter[j * 5 + i] * (float)input[col].x;
				sum.y += filter[j * 5 + i] * (float)input[col].y;
				sum.z += filter[j * 5 + i] * (float)input[col].z;
			}
			else if ((col >= 0) && (col < width) && (row >= height)) {
				sum.x += filter[j * 5 + i] * (float)input[col + stride * (height - 1)].x;
				sum.y += filter[j * 5 + i] * (float)input[col + stride * (height - 1)].y;
				sum.z += filter[j * 5 + i] * (float)input[col + stride * (height - 1)].z;
			}
		}
	}
	output[pos].x = (uchar)sum.x;
	output[pos].y = (uchar)sum.y;
	output[pos].z = (uchar)sum.z;
}

void ImageFilter::GaussianBlur(uchar3* input, uchar3* output, int w, int h, int s, int kernelsize) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	if (kernelsize == 3) {
		GaussianBlurKernel << < blocks, threads >> > (input, output, w, h, s, kernelsize);
	}
	else {
		GaussianBlur5Kernel << < blocks, threads >> > (input, output, w, h, s, kernelsize);
	}

}