#include "ImageFilter.h"

ImageFilter::ImageFilter() {
	this->BlockWidth = 32;
	this->BlockHeight = 12;
	this->StrideAlignment = 32;
}

ImageFilter::ImageFilter(int BlockWidth, int BlockHeight, int StrideAlignment) {
	this->BlockWidth = BlockWidth;
	this->BlockHeight = BlockHeight;
	this->StrideAlignment = StrideAlignment;
}

//Set-up memory in advance
int ImageFilter::initialize(int width, int height, int inputType, int outputType) {
	this->stride = iAlignUp(width);
	this->width = width;
	this->height = height;
	this->inputType = inputType;
	this->outputType = outputType;
	std::cout << "Image size processed: " << width << "x" << height << " (stride: " << stride << ")" << std::endl;
	std::cout << "Input Type: " << inputType << " || Output Type: " << outputType << std::endl;

	//parse input types
	if (inputType == CV_32F) {
		inputChannels = 1;
		inputDataSize = stride * height * sizeof(float);
		checkCudaErrors(cudaMalloc(&d_input32f, inputDataSize));
	}
	else if (inputType == CV_32FC2) {
		inputChannels = 2;
		inputDataSize = stride * height * sizeof(float) * 2;
		checkCudaErrors(cudaMalloc(&d_input32fc2, inputDataSize));
	}
	else if (inputType == CV_32FC3) {
		inputChannels = 3;
		inputDataSize = stride * height * sizeof(float) * 3;
		checkCudaErrors(cudaMalloc(&d_input32fc3, inputDataSize));
	}
	else if (inputType == CV_32FC4) {
		inputChannels = 4;
		inputDataSize = stride * height * sizeof(float) * 4;
		checkCudaErrors(cudaMalloc(&d_input32fc4, inputDataSize));
	}
	else if (inputType == CV_8U) {
		inputChannels = 1;
		inputDataSize = stride * height * sizeof(unsigned char);
		checkCudaErrors(cudaMalloc(&d_input8u, inputDataSize));
	}
	else if (inputType == CV_8UC2) {
		inputChannels = 2;
		inputDataSize = stride * height * sizeof(unsigned char) * 2;
		checkCudaErrors(cudaMalloc(&d_input8uc2, inputDataSize));
	}
	else if (inputType == CV_8UC3) {
		inputChannels = 3;
		inputDataSize = stride * height * sizeof(unsigned char) * 3;
		checkCudaErrors(cudaMalloc(&d_input8uc3, inputDataSize));
	}
	else if (inputType == CV_8UC4) {
		inputChannels = 4;
		inputDataSize = stride * height * sizeof(unsigned char) * 4;
		checkCudaErrors(cudaMalloc(&d_input8uc4, inputDataSize));
	}
	else {
		std::cout << "Input DataType not yet accepted. Use CV_8U-C1-C4 or CV_32F-C1-C4 only." << std::endl;
		return 1;
	}

	//parse output types
	if (outputType == CV_32F) {
		outputChannels = 1;
		outputDataSize = stride * height * sizeof(float);
		checkCudaErrors(cudaMalloc(&d_output32f, outputDataSize));
	}
	else if (outputType == CV_32FC2) {
		outputChannels = 2;
		outputDataSize = stride * height * sizeof(float) * 2;
	}
	else if (outputType == CV_32FC3) {
		outputChannels = 3;
		outputDataSize = stride * height * sizeof(float) * 3;
		checkCudaErrors(cudaMalloc(&d_output32fc3, outputDataSize));
	}
	else if (outputType == CV_32FC4) {
		outputChannels = 4;
		outputDataSize = stride * height * sizeof(float) * 4;
	}
	else if (outputType == CV_8U) {
		outputChannels = 1;
		outputDataSize = stride * height * sizeof(unsigned char);
	}
	else if (outputType == CV_8UC2) {
		outputChannels = 2;
		outputDataSize = stride * height * sizeof(unsigned char) * 2;
		checkCudaErrors(cudaMalloc(&d_output8uc2, outputDataSize));
	}
	else if (outputType == CV_8UC3) {
		outputChannels = 3;
		outputDataSize = stride * height * sizeof(unsigned char) * 3;
		checkCudaErrors(cudaMalloc(&d_output8uc3, outputDataSize));
	}
	else if (outputType == CV_8UC4) {
		outputChannels = 4;
		outputDataSize = stride * height * sizeof(unsigned char) * 4;
		checkCudaErrors(cudaMalloc(&d_output8uc4, outputDataSize));
	}
	else {
		std::cout << "Output DataType not yet accepted. Use CV_8U-C1-C4 or CV_32F-C1-C4 only." << std::endl;
		return 1;
	}

	return 0;
}

// General Purpos Filters
int ImageFilter::imfEdgeDetectHSV(cv::Mat input, cv::Mat &output, int kernelSize, float3 threshold) {
	if (inputType == CV_32FC3) {
		float3 *h_input32fc3 = (float3 *)input.ptr();
		checkCudaErrors(cudaMemcpy(d_input32fc3, h_input32fc3, inputDataSize, cudaMemcpyHostToDevice));
		EdgeDetectHsv(d_input32fc3, d_output32f, width, height, stride, kernelSize, threshold);
		checkCudaErrors(cudaMemcpy((float *)output.ptr(), d_output32f, outputDataSize, cudaMemcpyDeviceToHost));
	}
	else {
		std::cout << "Input channel type for HSV edge detection should be 32FC3." << std::endl;
		return 1;
	}
	return 0;
}

int ImageFilter::imfEdgeDetect(cv::Mat input, cv::Mat &output, int kernelSize, float threshold) {
	if (inputType == CV_8UC3) {
		uchar3* h_input8uc3 = (uchar3*)input.ptr();
		checkCudaErrors(cudaMemcpy(d_input8uc3, h_input8uc3, inputDataSize, cudaMemcpyHostToDevice));
		EdgeDetect(d_input8uc3, d_output32f, width, height, stride, kernelSize, threshold);
		checkCudaErrors(cudaMemcpy((float *)output.ptr(), d_output32f, outputDataSize, cudaMemcpyDeviceToHost));
	}
	else if (inputType == CV_32FC3) {
		float3 *h_input32fc3 = (float3 *)input.ptr();
		checkCudaErrors(cudaMemcpy(d_input32fc3, h_input32fc3, inputDataSize, cudaMemcpyHostToDevice));
		EdgeDetect(d_input32fc3, d_output32f, width, height, stride, kernelSize, threshold);
		checkCudaErrors(cudaMemcpy((float *)output.ptr(), d_output32f, outputDataSize, cudaMemcpyDeviceToHost));
	}
	else {
		std::cout << "Input channel type for edge detection is not yet implemented. Please use 8UC3." << std::endl;
		return 1;
	}
	return 0;
}

int ImageFilter::imfRgbToHsv(cv::Mat input, cv::Mat &output) {
	if (inputType == CV_8UC3) {
		uchar3* h_input8uc3 = (uchar3*)input.ptr();
		checkCudaErrors(cudaMemcpy(d_input8uc3, h_input8uc3, inputDataSize, cudaMemcpyHostToDevice));
		RgbToHsv(d_input8uc3, d_output32fc3, width, height, stride);
		checkCudaErrors(cudaMemcpy((float3 *)output.ptr(), d_output32fc3, outputDataSize, cudaMemcpyDeviceToHost));
	}
	else {
		std::cout << "Input and Output channels should match." << std::endl;
		return 1;
	}
	return 0;
}

int ImageFilter::imfUniform(cv::Mat input, cv::Mat &output, int kernelSize) {
	if (inputType == CV_8UC3) {
		uchar3* h_input8uc3 = (uchar3*)input.ptr();
		checkCudaErrors(cudaMemcpy(d_input8uc3, h_input8uc3, inputDataSize, cudaMemcpyHostToDevice));
		//for (int k = 1; k <= 1000; k++) { //for testing speed of convolution
		//	UniformFilter(d_input8uc3, d_output8uc3, width, height, stride, kernelSize);
		//}
		UniformFilter(d_input8uc3, d_output8uc3, width, height, stride, kernelSize);
		checkCudaErrors(cudaMemcpy((uchar3 *)output.ptr(), d_output8uc3, outputDataSize, cudaMemcpyDeviceToHost));
	}
	else {
		std::cout << "Input and Output channels should match." << std::endl;
		return 1;
	}
	return 0;
}

int ImageFilter::imfMedian(cv::Mat input, cv::Mat &output, int kernelSize) {
	if (inputType == CV_8UC3) {
		uchar3* h_input8uc3 = (uchar3*)input.ptr();
		std::cout << inputDataSize;
		checkCudaErrors(cudaMemcpy(d_input8uc3, h_input8uc3, 1242 * 375 * sizeof(uchar3), cudaMemcpyHostToDevice));
		//checkCudaErrors(cudaMemcpy(d_input8uc3, h_input8uc3, inputDataSize, cudaMemcpyHostToDevice));
		MedianFilter(d_input8uc3, d_output8uc3, width, height, stride, kernelSize);
		checkCudaErrors(cudaMemcpy((uchar3 *)output.ptr(), d_output8uc3, 1242 * 375 * sizeof(uchar3), cudaMemcpyDeviceToHost));
		//checkCudaErrors(cudaMemcpy((uchar3 *)output.ptr(), d_output8uc3, outputDataSize, cudaMemcpyDeviceToHost));
	}
	else {
		std::cout << "Input and Output channels should match." << std::endl;
		return 1;
	}
	return 0;
}

int ImageFilter::imfGaussianBlur(cv::Mat input, cv::Mat &output, int kernelSize) {
	if (inputType == CV_8UC3) {
		uchar3* h_input8uc3 = (uchar3*)input.ptr();
		checkCudaErrors(cudaMemcpy(d_input8uc3, h_input8uc3, inputDataSize, cudaMemcpyHostToDevice));
		GaussianBlur(d_input8uc3, d_output8uc3, width, height, stride, kernelSize);
		checkCudaErrors(cudaMemcpy((uchar3 *)output.ptr(), d_output8uc3, outputDataSize, cudaMemcpyDeviceToHost));
	}
	else {
		std::cout << "Input and Output channels should match." << std::endl;
		return 1;
	}
	return 0;
}


// Special Filters
int ImageFilter::imfCurveDetection(cv::Mat input, cv::Mat &output) {
	if (inputType == CV_8U) {
		uchar* h_input8u = (uchar*)input.ptr();
		checkCudaErrors(cudaMemcpy(d_input8u, h_input8u, inputDataSize, cudaMemcpyHostToDevice));
		CurveDetection(d_input8u, d_output32fc3, width, height, stride);
		checkCudaErrors(cudaMemcpy((float3 *)output.ptr(), d_output32fc3, outputDataSize, cudaMemcpyDeviceToHost));
	}
	else {
		std::cout << "Input and Output channels should match." << std::endl;
		return 1;
	}
	return 0;
}

int ImageFilter::imfMinMaxDetection(cv::Mat input, cv::Mat &output) {
	if (inputType == CV_8UC3) {
		uchar3* h_input8uc3 = (uchar3*)input.ptr();
		checkCudaErrors(cudaMemcpy(d_input8uc3, h_input8uc3, inputDataSize, cudaMemcpyHostToDevice));
		MinMaxDetection(d_input8uc3, d_output8uc3, width, height, stride);
		checkCudaErrors(cudaMemcpy((uchar3 *)output.ptr(), d_output8uc3, outputDataSize, cudaMemcpyDeviceToHost));
	}
	else {
		std::cout << "Input and Output channels should match." << std::endl;
		return 1;
	}
	return 0;
}


// OLD
int ImageFilter::imFilter(cv::Mat input, cv::Mat &output, const char *filterType, int kernelSize, float3 threshold) {
	////////////////////////////////////////////////////////////
	/// Edge Detection
	////////////////////////////////////////////////////////////
	if (strcmp(filterType, "edgedetect_hsv") == 0) {
		if (inputType == CV_32FC3) {
			float3 *h_input32fc3 = (float3 *)input.ptr();
			checkCudaErrors(cudaMemcpy(d_input32fc3, h_input32fc3, inputDataSize, cudaMemcpyHostToDevice));
			EdgeDetectHsv(d_input32fc3, d_output32f, width, height, stride, kernelSize, threshold);
			checkCudaErrors(cudaMemcpy((float *)output.ptr(), d_output32f, outputDataSize, cudaMemcpyDeviceToHost));
		}
		else {
			std::cout << "Input channel type for HSV edge detection should be 32FC3." << std::endl;
			return 1;
		}
	}
	return 0;
}

int ImageFilter::imFilter(cv::Mat input, cv::Mat &output, const char *filterType, int kernelSize, float threshold) {
	////////////////////////////////////////////////////////////
	/// Edge Detection
	///////////////////////////////////////////////////////////
	if ((strcmp(filterType, "edgedetect") == 0) || (strcmp(filterType, "edgedetect_rgb") == 0)) {
		if (inputType == CV_8UC3) {
			uchar3* h_input8uc3 = (uchar3*)input.ptr();
			checkCudaErrors(cudaMemcpy(d_input8uc3, h_input8uc3, inputDataSize, cudaMemcpyHostToDevice));
			EdgeDetect(d_input8uc3, d_output32f, width, height, stride, kernelSize, threshold);
			checkCudaErrors(cudaMemcpy((float *)output.ptr(), d_output32f, outputDataSize, cudaMemcpyDeviceToHost));
		}
		else if (inputType == CV_32FC3) {
			float3 *h_input32fc3 = (float3 *)input.ptr();
			checkCudaErrors(cudaMemcpy(d_input32fc3, h_input32fc3, inputDataSize, cudaMemcpyHostToDevice));
			EdgeDetect(d_input32fc3, d_output32f, width, height, stride, kernelSize, threshold);
			checkCudaErrors(cudaMemcpy((float *)output.ptr(), d_output32f, outputDataSize, cudaMemcpyDeviceToHost));
		}
		else {
			std::cout << "Input channel type for edge detection is not yet implemented. Please use 8UC3." << std::endl;
			return 1;
		}
	}
	return 0;
}

int ImageFilter::imFilter(cv::Mat input, cv::Mat &output, const char *filterType, int kernelSize) {
	//depending on the filter-type, the input and output channels
	////////////////////////////////////////////////////////
	/// RGB to HSV 
	////////////////////////////////////////////////////////
	if (strcmp(filterType, "rgbtohsv") == 0) {
		//inputchannel = outputchannel
		if (inputType == CV_8UC3) {
			uchar3* h_input8uc3 = (uchar3*)input.ptr();
			checkCudaErrors(cudaMemcpy(d_input8uc3, h_input8uc3, inputDataSize, cudaMemcpyHostToDevice));
			RgbToHsv(d_input8uc3, d_output32fc3, width, height, stride);
			checkCudaErrors(cudaMemcpy((float3 *)output.ptr(), d_output32fc3, outputDataSize, cudaMemcpyDeviceToHost));
		}
		else {
			std::cout << "Input and Output channels should match." << std::endl;
			return 1;
		}
	}

	////////////////////////////////////////////////////////
	/// Uniform Filter
	////////////////////////////////////////////////////////
	if (strcmp(filterType, "box") == 0) {
		//inputchannel = outputchannel

		if (inputType == CV_8UC3) {
			uchar3* h_input8uc3 = (uchar3*)input.ptr();
			checkCudaErrors(cudaMemcpy(d_input8uc3, h_input8uc3, inputDataSize, cudaMemcpyHostToDevice));
			UniformFilter(d_input8uc3, d_output8uc3, width, height, stride, kernelSize);
			checkCudaErrors(cudaMemcpy((uchar3 *)output.ptr(), d_output8uc3, outputDataSize, cudaMemcpyDeviceToHost));
		}
		else {
			std::cout << "Input and Output channels should match." << std::endl;
			return 1;
		}
	}

	/////////////////////////////////////////////////////////
	/// Median Filter
	/////////////////////////////////////////////////////////
	else if (strcmp(filterType, "median") == 0) {
		//inputchannel = outputchannel
		if (inputType == CV_8UC3) {
			uchar3* h_input8uc3 = (uchar3*)input.ptr();
			checkCudaErrors(cudaMemcpy(d_input8uc3, h_input8uc3, inputDataSize, cudaMemcpyHostToDevice));
			MedianFilter(d_input8uc3, d_output8uc3, width, height, stride, kernelSize);
			checkCudaErrors(cudaMemcpy((uchar3 *)output.ptr(), d_output8uc3, outputDataSize, cudaMemcpyDeviceToHost));
		}
		else {
			std::cout << "Input and Output channels should match." << std::endl;
			return 1;
		}
	}

	////////////////////////////////////////////////////////////
	/// Gaussian Blur
	////////////////////////////////////////////////////////////
	else if (strcmp(filterType, "gaussianblur") == 0) {
		if (inputType == CV_8UC3) {
			uchar3* h_input8uc3 = (uchar3*)input.ptr();
			checkCudaErrors(cudaMemcpy(d_input8uc3, h_input8uc3, inputDataSize, cudaMemcpyHostToDevice));
			GaussianBlur(d_input8uc3, d_output8uc3, width, height, stride, kernelSize);
			checkCudaErrors(cudaMemcpy((uchar3 *)output.ptr(), d_output8uc3, outputDataSize, cudaMemcpyDeviceToHost));
		}
		else {
			std::cout << "Input and Output channels should match." << std::endl;
			return 1;
		}
	}



	return 0;
}
// END OF OLD

int ImageFilter::close() {
	if (inputType == CV_8U) {
		checkCudaErrors(cudaFree(d_input8u));
	}
	if (inputType == CV_8UC3) {
		checkCudaErrors(cudaFree(d_input8uc3));
	}
	if (inputType == CV_32FC3) {
		checkCudaErrors(cudaFree(d_input32fc3));
	}
	if (outputType == CV_8UC3) {
		checkCudaErrors(cudaFree(d_output8uc3));
	}
	if (outputType == CV_32F) {
		checkCudaErrors(cudaFree(d_output32f));
	}
	if (outputType == CV_32FC3) {
		checkCudaErrors(cudaFree(d_output32fc3));
	}
	return 0;
}

inline int ImageFilter::iAlignUp(int n)
{
	int m = this->StrideAlignment;
	int mod = n % m;

	if (mod)
		return n + m - mod;
	else
		return n;
}

// swap two values
template<typename T>
inline void ImageFilter::Swap(T &a, T &b)
{
	T t = a;
	a = b;
	b = t;
}

