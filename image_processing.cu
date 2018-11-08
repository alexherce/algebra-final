#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <omp.h>

using namespace std;

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

__global__ void grayscale_filter_gpu(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep, int grayWidthStep)
{
	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Only valid threads perform memory I/O
	if ((xIndex < width) && (yIndex < height))
	{
		//Location of colored pixel in input
		const int color_tid = yIndex * colorWidthStep + (3 * xIndex);

		//Location of gray pixel in output
		const int gray_tid = yIndex * grayWidthStep + xIndex;

		const unsigned char blue = input[color_tid];
		const unsigned char green = input[color_tid + 1];
		const unsigned char red = input[color_tid + 2];

		const float gray = (red + green + blue) / 3.f;

		output[gray_tid] = static_cast<unsigned char>(gray);
	}
}

__global__ void negative_filter_gpu(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep)
{
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

	int color_tid = (yIndex * width + xIndex) * 3;

	const unsigned char blue = -1 * input[color_tid];
	const unsigned char green = -1 * input[color_tid + 1];
	const unsigned char red = -1 * input[color_tid + 2];

	output[color_tid] = static_cast<unsigned char>(blue);
	output[color_tid + 1] = static_cast<unsigned char>(green);
	output[color_tid + 2] = static_cast<unsigned char>(red);
}

__global__ void sepia_filter_gpu(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep)
{
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

	int color_tid = (yIndex * width + xIndex) * 3;

	const unsigned char blue = input[color_tid];
	const unsigned char green = input[color_tid + 1];
	const unsigned char red = input[color_tid + 2];

	// tr = 0.393R + 0.769G + 0.189B
	// tg = 0.349R + 0.686G + 0.168B
	// tb = 0.272R + 0.534G + 0.131B

	unsigned int red_s = (int)(0.393 * red + 0.769 * green + 0.189 * blue);
	unsigned int green_s = (int)(0.349 * red + 0.686 * green + 0.168 * blue);
	unsigned int blue_s = (int)(0.272 * red + 0.534 * green + 0.131 * blue);

	// Verify that color is min 0 and max 255
	if (red_s > 255) red_s = 255;
	if (green_s > 255) green_s = 255;
	if (blue_s > 255) blue_s = 255;

	output[color_tid] = static_cast<unsigned char>(blue_s);
	output[color_tid + 1] = static_cast<unsigned char>(green_s);
	output[color_tid + 2] = static_cast<unsigned char>(red_s);
}

__global__ void contrast_filter_gpu(unsigned char* input, unsigned char* output, int width, int height, int contrast)
{
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

	int color_tid = (yIndex * width + xIndex) * 3;

	const unsigned char blue = input[color_tid];
	const unsigned char green = input[color_tid + 1];
	const unsigned char red = input[color_tid + 2];

	float factor = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast));

	unsigned int red_s = (int)(factor * (red - 128) + 128);
	unsigned int green_s = (int)(factor * (green - 128) + 128);
	unsigned int blue_s = (int)(factor * (blue - 128) + 128);

	// Verify that color is min 0 and max 255
	if (red_s < 0) red_s = 0;
	if (red_s > 255) red_s = 255;
	if (green_s < 0) green_s = 0;
	if (green_s > 255) green_s = 255;
	if (blue_s < 0) blue_s = 0;
	if (blue_s > 255) blue_s = 255;

	output[color_tid] = static_cast<unsigned char>(blue_s);
	output[color_tid + 1] = static_cast<unsigned char>(green_s);
	output[color_tid + 2] = static_cast<unsigned char>(red_s);
}

__global__ void get_histogram(unsigned char* input, float* histogram, int width, int height, int step) {
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	const int index = yIndex * step + xIndex;

	if (xIndex < width && yIndex < height) {
		atomicAdd(&histogram[input[index]], 1);
	}
}

__global__ void normalize_histogram(float* histogram, float* histogram_normalized, int width, int height) {

	unsigned int nxy = threadIdx.x + threadIdx.y * blockDim.x;

	if (nxy < 256 && blockIdx.x == 0 && blockIdx.y == 0) {
		for (int i = 0; i < nxy; i++) {
			histogram_normalized[nxy] += histogram[i];
		}
		histogram_normalized[nxy] = histogram_normalized[nxy] * 255 / (width * height);
	}

}

__global__ void apply_histogram_image(unsigned char* input, unsigned char* output, float* histogram_normalized, int width, int height, int step) {
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	const int index = yIndex * step + xIndex;

	if (xIndex < width && yIndex < height) {
		output[index] = histogram_normalized[input[index]];
	}
}

__global__ void blur_kernel(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep) {

	//2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if ((xIndex < width) && (yIndex < height)) {

		// Location of pixel in output
		const int outputIndex = yIndex * colorWidthStep + (xIndex * 3);
		int blue = 0;
		int green = 0;
		int red = 0;
		int det = 0;

		// Iterate filter matrix
		for (int x = -2; x < 3; x++) {
			for (int y = -2; y < 3; y++) {

				int inputIndex = (y + yIndex) * colorWidthStep + ((x + xIndex) * 3);

				// Check if it is inside borders
				if ((xIndex + x < width) && (yIndex + y < height) && (xIndex + x > 0) && (yIndex + y > 0)) {
					blue += input[inputIndex];
					green += input[inputIndex + 1];
					red += input[inputIndex + 2];
					det++;
				}
			}
		}

		// Store in output. Divide between number of iterations (ideally divide by 25 since its a 5x5 filter)
		output[outputIndex] = static_cast<unsigned char>(blue / det);
		output[outputIndex + 1] = static_cast<unsigned char>(green / det);
		output[outputIndex + 2] = static_cast<unsigned char>(red / det);
	}
}


// FUNCTIONS

void grayscale_filter(const cv::Mat& input)
{
	cv::Mat output(input.rows, input.cols, CV_8UC1); //Grayscale image

	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;
	// Calculate total number of bytes of input and output image
	// Step = cols * number of colors	
	size_t colorBytes = input.step * input.rows;
	size_t grayBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, grayBytes), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(16, 16);

	// Calculate grid size to cover the whole image
	// const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows / block.y));
	printf("grayscale_filter_gpu<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

	// Launch the color conversion kernel
	grayscale_filter_gpu << <grid, block >> > (d_input, d_output, input.cols, input.rows, static_cast<int>(input.step), static_cast<int>(output.step));

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");

	// Save Image
	cv::imwrite("car_grayscale.jpg", output);

	/* ********* DISPLAY IMAGE **********/
	//Allow the windows to resize
	namedWindow("GPU GRAYSCALE", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("GPU GRAYSCALE", output);
}

void negative_filter(const cv::Mat& input)
{
	cv::Mat output(input.rows, input.cols, input.type()); // Color image

	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;
	// Calculate total number of bytes of input and output image
	// Step = cols * number of colors	
	size_t colorBytes = input.step * input.rows;

	unsigned char *d_input, *d_output;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, colorBytes), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(16, 16);

	// Calculate grid size to cover the whole image
	// const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows / block.y));
	printf("negative_filter_gpu<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

	// Launch the color conversion kernel
	negative_filter_gpu << <grid, block >> > (d_input, d_output, input.cols, input.rows, colorBytes);

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, colorBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");

	// Save Image
	cv::imwrite("car_negative.jpg", output);

	/* ********* DISPLAY IMAGE **********/
	//Allow the windows to resize
	namedWindow("GPU NEGATIVE", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("GPU NEGATIVE", output);
}

void sepia_filter(const cv::Mat& input)
{
	cv::Mat output(input.rows, input.cols, input.type()); // Color image

	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;
	// Calculate total number of bytes of input and output image
	// Step = cols * number of colors	
	size_t colorBytes = input.step * input.rows;

	unsigned char *d_input, *d_output;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, colorBytes), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(16, 16);

	// Calculate grid size to cover the whole image
	// const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows / block.y));
	printf("sepia_filter_gpu<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

	// Launch the color conversion kernel
	sepia_filter_gpu << <grid, block >> > (d_input, d_output, input.cols, input.rows, colorBytes);

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, colorBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");

	// Save Image
	cv::imwrite("car_sepia.jpg", output);

	/* ********* DISPLAY IMAGE **********/
	//Allow the windows to resize
	namedWindow("GPU SEPIA", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("GPU SEPIA", output);
}

void contrast_filter(const cv::Mat& input)
{
	cv::Mat output(input.rows, input.cols, input.type()); // Color image

	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;
	// Calculate total number of bytes of input and output image
	// Step = cols * number of colors	
	size_t colorBytes = input.step * input.rows;

	unsigned char *d_input, *d_output;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, colorBytes), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(16, 16);

	// Calculate grid size to cover the whole image
	// const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows / block.y));
	printf("contrast_filter_gpu<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

	// Launch the color conversion kernel
	contrast_filter_gpu << <grid, block >> > (d_input, d_output, input.cols, input.rows, -128);

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, colorBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");

	// Save Image
	cv::imwrite("car_contrast.jpg", output);

	/* ********* DISPLAY IMAGE **********/
	//Allow the windows to resize
	namedWindow("GPU CONTRAST", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("GPU CONTRAST", output);
}

void blur_image(const cv::Mat& input)
{
	cv::Mat output(input.rows, input.cols, input.type()); // Color image

	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;

	// Calculate total number of bytes of input and output image
	// Step = cols * number of colors	
	size_t inputBytes = input.step * input.rows;
	size_t outputBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, inputBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, outputBytes), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), inputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(32, 32);

	// Calculate grid size to cover the whole image
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows / block.y));

	//const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
	//const dim3 grid(16, 4);

	printf("blur_kernel<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

	// Start timer
	auto start_gpu = chrono::high_resolution_clock::now();

	// Launch the color conversion kernel
	blur_kernel << <grid, block >> > (d_input, d_output, input.cols, input.rows, static_cast<int>(input.step));

	// End timer and print result
	auto end_gpu = chrono::high_resolution_clock::now();
	chrono::duration<float, std::milli> duration_ms = end_gpu - start_gpu;
	printf("GPU elapsed %f ms\n", duration_ms.count());

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, outputBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");

	// Save Image
	cv::imwrite("car_blurred.jpg", output);

	/* ********* DISPLAY IMAGE **********/
	//Allow the windows to resize
	namedWindow("GPU BLURRED", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("GPU BLURRED", output);
}

void normalize_gpu(const cv::Mat& color_input) {

	//Create output image
	cv::Mat input;
	cvtColor(color_input, input, cv::COLOR_BGR2GRAY);

	//creating output image
	cv::Mat output(input.rows, input.cols, input.type());

	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;

	// Calculate total number of bytes of input and output image
	// Step = cols * number of colors	
	size_t inputBytes = input.step * input.rows;
	size_t outputBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;
	float * d_histogram = {};
	float * d_histogram_normalized = {};

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, inputBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, outputBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc(&d_histogram, 256 * sizeof(float)), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc(&d_histogram_normalized, 256 * sizeof(float)), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), inputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_output, output.ptr(), outputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(32, 32);

	// Calculate grid size to cover the whole image
	// const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
	const dim3 grid((input.cols) / block.x, (input.rows) / block.y);

	// Launch the color conversion kernel
	auto start_cpu = chrono::high_resolution_clock::now();
	get_histogram << <grid, block >> > (d_input, d_histogram, input.cols, input.rows, static_cast<int>(input.step));
	normalize_histogram << <grid, block >> > (d_histogram, d_histogram_normalized, input.cols, input.rows);
	apply_histogram_image << <grid, block >> > (d_input, d_output, d_histogram_normalized, input.cols, input.rows, static_cast<int>(input.step));
	auto end_cpu = chrono::high_resolution_clock::now();

	chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
	printf("GPU elapsed %f ms\n", duration_ms.count());

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, outputBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");

	// Save Image
	cv::imwrite("histogram.jpg", input);
	cv::imwrite("histogram_normalized.jpg", output);

	/* ********* DISPLAY IMAGE **********/
	//Allow the windows to resize
	namedWindow("GPU HISTOGRAM NORMALIZED", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("GPU HISTOGRAM NORMALIZED", output);
}

cv::Mat load_image(string imagePath) {

	cout << "Image: " << imagePath << endl;
	// Read input image from the disk
	cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

	return input;
}

int main(int argc, char *argv[])
{
	// Set up GPU device
	int dev = 0;
	cudaDeviceProp deviceProp;
	SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	SAFE_CALL(cudaSetDevice(dev), "Error setting device");

	cv::Mat input = load_image("car.jpg");

	if (input.empty())
	{
		cout << "Image Not Found!" << endl;
		cin.get();
		return -1;
	}

	int selected_option = 0;

	cout << "Seleccione una opcion para aplicar a la imagen:" << endl;
	cout << "1.- Filtro de escala de grises" << endl;
	cout << "2.- Filtro negativo" << endl;
	cout << "3.- Filtro sepia" << endl;
	cout << "4.- Filtro de contraste" << endl;
	cout << "5.- Filtro de blur" << endl;
	cout << "6.- Normalizar histograma" << endl;
	cout << "7.- Salir" << endl;

	cin >> selected_option;

	switch (selected_option) {
		case 1:
			grayscale_filter(input);
			break;

		case 2:
			negative_filter(input);
			break;

		case 3:
			sepia_filter(input);
			break;

		case 4:
			contrast_filter(input);
			break;

		case 5:
			blur_image(input);
			break;

		case 6: {
			input = load_image("Images/dog1.jpeg");

			if (input.empty())
			{
				cout << "Image Not Found!" << endl;
				cin.get();
				break;
			}

			normalize_gpu(input);
			break;
		}

		case 7:
			break;

		default:
			cout << "Opcion no reconocida.";
	}

	/* ********* DISPLAY IMAGE **********/
	//Allow the windows to resize
	namedWindow("GPU INPUT", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("GPU INPUT", input);

	//Wait for key press
	cv::waitKey();

	return 0;
}
