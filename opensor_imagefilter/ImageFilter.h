#ifndef IMAGEFILTER_H
#define IMAGEFILTER_H

#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <memory.h>
#include <math.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>

#include "common.h"
#include "liblink.h"

class ImageFilter {
public:
	ImageFilter();
	ImageFilter(int BlockWidth, int BlockHeight, int StrideAlignment);
	~ImageFilter() {};
	//////////////////////////////////////////////////////////////////////////////////////
	/// GENERAL PURPOSE FILTERS
	//////////////////////////////////////////////////////////////////////////////////////
	//Functions for General Purpose Filters
	int initialize(int width, int height, int inputType, int outputType);
	int imfEdgeDetectHSV(cv::Mat input, cv::Mat &output, int kernelSize, float3 threshold);
	int imfEdgeDetect(cv::Mat input, cv::Mat &output, int kernelSize, float threshold);
	int imfRgbToHsv(cv::Mat input, cv::Mat &output);
	int imfUniform(cv::Mat input, cv::Mat &output, int kernelSize);
	int imfMedian(cv::Mat input, cv::Mat &output, int kernelSize);
	int imfGaussianBlur(cv::Mat input, cv::Mat &output, int kernelSize);
	int imfCurveDetection(cv::Mat input, cv::Mat &output);
	int imfMinMaxDetection(cv::Mat input, cv::Mat &output);


	///OLD
	int imFilter(cv::Mat input, cv::Mat &output, const char *filterType, int kernelSize);
	int imFilter(cv::Mat input, cv::Mat &output, const char *filterType, int kernelSize, float threshold);
	int imFilter(cv::Mat input, cv::Mat &output, const char *filterType, int kernelSize, float3 threshold);
	int close();
	inline int iAlignUp(int n);
	template<typename T> inline void Swap(T &a, T &b);

	//general purpose filters
	void RgbToHsv(uchar3* input, float3* output, int w, int h, int s);
	void UniformFilter(uchar3* input, uchar3* output, int w, int h, int s, int kernelsize);
	void MedianFilter(uchar3* input, uchar3* output, int w, int h, int s, int kernelsize);
	void EdgeDetect(uchar3* input, float* output, int w, int h, int s, int kernelsize, float threshold);
	void EdgeDetect(float3* input, float* output, int w, int h, int s, int kernelsize, float threshold);
	void EdgeDetectHsv(float3* input, float* output, int w, int h, int s, int kernelsize, float3 threshold);
	void GaussianBlur(uchar3* input, uchar3* output, int w, int h, int s, int kernelsize);

	//Special filters and processing
	void CurveDetection(uchar* input, float3* output, int w, int h, int s);
	void MinMaxDetection(uchar3* input, uchar3* output, int width, int height, int stride);

	int BlockHeight, BlockWidth, StrideAlignment;
	//template <typename T> T d_input;
	//template <typename T> T d_output;

	float4 *d_input32fc4;
	float4 *d_output32fc4;
	float3 *d_input32fc3;
	float3 *d_output32fc3;
	float2 *d_input32fc2;
	float2 *d_output32fc2;
	float *d_input32f;
	float *d_output32f;

	uchar4 *d_input8uc4;
	uchar4 *d_output8uc4;
	uchar3 *d_input8uc3;
	uchar3 *d_output8uc3;
	uchar2 *d_input8uc2;
	uchar2 *d_output8uc2;
	uchar *d_input8u;
	uchar *d_output8u;

	int width;
	int height;
	int stride;
	int inputDataSize;
	int outputDataSize;
	int inputType; //CV types
	int outputType;
	int inputChannels;
	int outputChannels;
};

#endif