
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>

void readCSV();
void generateFrameOffsets();
void generateFilter();
std::vector<double> windowData(int frameOffset);
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

const char delimeter = ',';//delimeter between items in CSV file
std::vector<double>inputArray;//array of input data from CSV file
int inputArraySize = 0;
std::vector<int>frameOffsets;//list of frame offsets used by workers
std::vector<double>filter;//filter used to window the FFT frames


//user set parameters
const int k_fftInputLen = 100; //length of FFT input array
const int k_fftFrameOffset = 10; //offset between start of FFT frames(eg x[n]=x[n-1]+k_fftFrameOffset where x[n] is the first value used as input to the fft frame)

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    //Todo need to review garbage collection and do prototype until simple FFT is complete

    std::cout << "Start of program\n";
    readCSV();

    //generate blackman-harris filter from 0 to k_fftInputLen-1
    generateFilter();

    //generate frameOffsets
    generateFrameOffsets();

    //todo fxns below will be run in parallel

    std::vector<double> window = windowData(0);

    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

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

void readCSV()
{
    std::string line;
    std::string string;

    std::ifstream myFile("data.CSV");

    if (!myFile.is_open()) throw std::runtime_error("Couldn't open file");

    while (getline(myFile, string, delimeter)) {
        //std::cout << string << std::endl;
        inputArray.push_back(std::stod(string));
        inputArraySize++;
    }
}

void generateFrameOffsets()
{
    for (int i = 0; i < (inputArraySize - k_fftInputLen); i += k_fftFrameOffset)
    {
        frameOffsets.push_back(i);
    }
}

void generateFilter()
{
    //w[n] = a0 - a1*cos(x) + a2*cos(2x) - a3cos(3x), x = (2n*pi)/N, 0 < n < N

    const double a0 = 0.35875;
    const double a1 = 0.48829;
    const double a2 = 0.14128;
    const double a3 = 0.01168;

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

        filter.push_back(w_n);
    }
}

/*  Apply blackman - harris filter to input data frame.
 *  Return the result as a vector.                      */
std::vector<double> windowData(int frameOffset)
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

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
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
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

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
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
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
