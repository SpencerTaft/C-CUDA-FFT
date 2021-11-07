﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <complex>
#include <iostream>
#include <valarray>

/**********************************************************
 * Declarations
 **********************************************************/
//Type definitions
typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;

//Function declarations
std::vector<double> readCSV();
std::vector<int> generateFrameOffsets();
std::vector<double> generateFilter();
std::vector<double> windowData(int frameOffset, std::vector<double> filter);
void fft(CArray& x);
cudaError_t addWithCuda();
cudaError_t FFTWithCuda(std::vector<double>& filter, std::vector<int>& frameOffsets, std::vector<double>& inputArray);

//Global variables
int inputArraySize = 0;

//Global FFT variables
const double PI = 3.141592653589793238460;
typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;

//user set parameters
const int k_fftInputLen = 100; //length of FFT input array
const int k_fftFrameOffset = 10; //offset between start of FFT frames(eg x[n]=x[n-1]+k_fftFrameOffset where x[n] is the first value used as input to the fft frame)

/**********************************************************
 * Functions run on single thread
 **********************************************************/
std::vector<double> readCSV()
{
    std::vector<double> returnVector;

    const char delimeter = ',';//delimeter between items in CSV file
    std::string line;
    std::string string;

    std::ifstream myFile("data.CSV");

    if (!myFile.is_open()) throw std::runtime_error("Couldn't open file");

    while (getline(myFile, string, delimeter)) {
        returnVector.push_back(std::stod(string));
        inputArraySize++;
    }

    return returnVector;
}

std::vector<int> generateFrameOffsets()
{
    std::vector<int> offsets;

    for (int i = 0; i < (inputArraySize - k_fftInputLen); i += k_fftFrameOffset)
    {
        offsets.push_back(i);
    }

    return offsets;
}

std::vector<double> generateFilter()
{
    //w[n] = a0 - a1*cos(x) + a2*cos(2x) - a3cos(3x), x = (2n*pi)/N, 0 < n < N
    const double a0 = 0.35875;
    const double a1 = 0.48829;
    const double a2 = 0.14128;
    const double a3 = 0.01168;

    std::vector<double> outputFilter;
    double x;
    double term1, term2, term3;
    double w_n;

    for (int n = 0; n < k_fftInputLen; n++)
    {
        //calculate x
        x = 2 * n * (3.14159);
        x /= k_fftInputLen;

        term1 = a1 * cos(x);
        term2 = a2 * cos(2 * x);
        term3 = a3 * cos(3 * x);

        w_n = a0 - term1 + term2 - term3;

        outputFilter.push_back(w_n);
    }

    return outputFilter;
}

 /**********************************************************
  * Functions run in parallel
  **********************************************************/
  /*  Apply blackman - harris filter to input data frame.
   *  Return the result as a vector.                      */
std::vector<double> windowData(int frameOffset, std::vector<double> filter, std::vector<double> inputArray)
{
    std::vector<double>windowedVector;
    double windowedData;

    for (int n = 0; n < k_fftInputLen; n++)
    {
        windowedData = filter[n] * inputArray[n + frameOffset];
        windowedVector.push_back(windowedData);
    }

    return windowedVector;
}

// Cooley–Tukey FFT (in-place)
void fft(CArray& x)
{
    const size_t N = x.size();
    if (N <= 1) return;

    // divide
    CArray even = x[std::slice(0, N / 2, 2)];
    CArray  odd = x[std::slice(1, N / 2, 2)];

    // conquer
    fft(even);
    fft(odd);

    // combine
    for (size_t k = 0; k < N / 2; ++k)
    {
        Complex t = std::polar(1.0, -2 * PI * k / N) * odd[k];
        x[k] = even[k] + t;         //todo this is overwriting the input data, need to preserve input copy
        x[k + N / 2] = even[k] - t; //todo this is overwriting the input data, need to preserve input copy
    }
}

 /**********************************************************
  * Main
  **********************************************************/
int main()
{
    std::cout << "Start of program\n";

    //Read CSV file and put elements in inputArray vector
    std::vector<double> inputArray = readCSV();

    //generate blackman-harris filter from 0 to k_fftInputLen-1 to window the input data
    std::vector<double> filter = generateFilter();

    //generate frameOffsets
    std::vector<int> frameOffsets = generateFrameOffsets();//list of frame offsets used by workers

    //todo fxns below will be run in parallel

    std::vector<double> windowedData = windowData(0, filter, inputArray);

    Complex test[k_fftInputLen];
    for (int i = 0; i < k_fftInputLen; i++)
    {
        test[i] = windowedData[i];
    }

    CArray data(test, k_fftInputLen);
    
    // forward fft
    fft(data);

    std::cout << "fft" << std::endl;
    for (int i = 0; i < 8; ++i)
    {
        std::cout << data[i] << std::endl;
    }

    // Run FFT in parallel.
    cudaError_t cudaStatus = FFTWithCuda(filter, frameOffsets, inputArray);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "FFTWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    std::cout << "End of program\n";

    return 0;
}

//((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((
//((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((
//((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((
//((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((
cudaError_t FFTWithCuda(std::vector<double>& filter, std::vector<int>& frameOffsets, std::vector<double>& inputArray)
{
    cudaError_t cudaStatus;
    int* dev_filter = 0;
    int* dev_frameOffsets = 0;
    int* dev_inputArray = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    //Allocate space on GPU for the filter, frameOffsets, and inputArray.  Only one of each is needed

    cudaStatus = cudaMalloc((void**)&dev_filter, sizeof(filter));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_frameOffsets, sizeof(frameOffsets));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_inputArray, sizeof(inputArray));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    //Copy input vectors from host memory to GPU buffers.

    
    cudaStatus = cudaMemcpy(dev_filter, &filter[0], sizeof(filter), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_frameOffsets, &frameOffsets[0], sizeof(frameOffsets), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_inputArray, &inputArray[0], sizeof(inputArray), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_filter);
    cudaFree(dev_frameOffsets);
    cudaFree(dev_inputArray);

    return cudaStatus;
}
//))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
//))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
//))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
//))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))



//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda()
{
    const int arraySize = 5;
    unsigned int size = arraySize;//todo var is redundant


    int a_local[arraySize] = { 1, 2, 3, 4, 5 };
    int* cpu_a = &a_local[0];

    int b_local[arraySize] = { 10, 20, 30, 40, 50 };
    int* cpu_b = &b_local[0];

    int c_local[arraySize] = { 0 };
    int* cpu_c = &c_local[0];

    //c, a, b, arraySize


    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, cpu_a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, cpu_b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(cpu_c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<