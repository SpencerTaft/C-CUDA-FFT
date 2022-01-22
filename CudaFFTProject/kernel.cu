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

template <class T> class ContiguousArray
{
public:
    unsigned int numElements;
    T* ptr;

    ContiguousArray(T* inputPtr, unsigned int inputNumElements)
    {
        ptr = inputPtr;
        numElements = inputNumElements;
    }

    ContiguousArray()
    {
        ptr = nullptr;
        numElements = 0;
    }

    unsigned int getSize()
    {
        return numElements * sizeof(T);
    }
};

typedef struct Comp
{
    float real;
    float imag;
};

//Function declarations
ContiguousArray<float> readCSV();
ContiguousArray<int> generateFrameOffsets();
ContiguousArray<float> generateFilter();
std::vector<float> windowData(int frameOffset, std::vector<float> filter);
void fft(CArray& x);
cudaError_t FFTWithCuda(ContiguousArray<float> filter, ContiguousArray<int> frameOffsets, ContiguousArray <float> inputArray);

//Global variables
int inputArraySize = 0;

//Global FFT variables
//const float PI = (float)3.141592653589793238460;
//typedef std::complex<float> Complex;
//typedef std::valarray<Complex> CArray;

//user set parameters
const int k_fftInputLen = 512; //length of FFT input array(data points per FFT frame)
const int k_fftFrameOffset = 10; //offset between start of FFT frames(eg x[n]=x[n-1]+k_fftFrameOffset where x[n] is the first value used as input to the fft frame)

/**********************************************************
 * Functions run on single thread
 **********************************************************/
ContiguousArray<float> readCSV()
{
    std::vector<float> csvVector;
    ContiguousArray<float> retArray;

    const char delimeter = ',';//delimeter between items in CSV file
    std::string line;
    std::string string;

    std::ifstream myFile("data.CSV");

    if (!myFile.is_open()) throw std::runtime_error("Couldn't open file");

    while (getline(myFile, string, delimeter)) {
        csvVector.push_back(std::stof(string));
        inputArraySize++;
    }

    //CUDA requires contiguous memory
    retArray.numElements = inputArraySize;
    retArray.ptr = new float[retArray.numElements];

    for (int i = 0; i < inputArraySize; i++)
    {
        retArray.ptr[i] = csvVector[i];
    }

    return retArray;
}

ContiguousArray<int> generateFrameOffsets()
{
    ContiguousArray<int> offsets;

    //CUDA requires contiguous memory
    offsets.numElements = (inputArraySize - k_fftInputLen) / k_fftFrameOffset;
    offsets.ptr = new int[offsets.numElements];

    for (int i = 0; (i * k_fftFrameOffset) <= (inputArraySize - k_fftInputLen); i++)
    {
        offsets.ptr[i] = (i * k_fftFrameOffset);
    }

    return offsets;
}

ContiguousArray<float> generateFilter()
{
    //w[n] = a0 - a1*cos(x) + a2*cos(2x) - a3cos(3x), x = (2n*pi)/N, 0 < n < N
    const float a0 = 0.35875f;
    const float a1 = 0.48829f;
    const float a2 = 0.14128f;
    const float a3 = 0.01168f;

    ContiguousArray<float> retArray;
    float x;
    float term1, term2, term3;
    float w_n;

    retArray.numElements = k_fftInputLen;
    retArray.ptr = new float[retArray.numElements];

    for (int n = 0; n < k_fftInputLen; n++)
    {
        //calculate x
        x = 2 * n * (3.14159);
        x /= k_fftInputLen;

        term1 = a1 * cos(x);
        term2 = a2 * cos(2 * x);
        term3 = a3 * cos(3 * x);

        w_n = a0 - term1 + term2 - term3;

        retArray.ptr[n] = w_n;
    }

    return retArray;
}

 /**********************************************************
  * Functions run in parallel
  **********************************************************/

/*  Apply blackman - harris filter to input data frame.
 *  Return the result as a vector.                      */
std::vector<float> windowData(int frameOffset, std::vector<float> filter, std::vector<float> inputArray)
{
    std::vector<float>windowedVector;
    float windowedData;

    for (int n = 0; n < k_fftInputLen; n++)
    {
        windowedData = filter[n] * inputArray[n + frameOffset];
        windowedVector.push_back(windowedData);
    }

    return windowedVector;
}

__device__ void kernelWindowData(int frameOffset, float* filterVec, float* inputArrayVec, Comp* windowedDataI)
{
    for (int n = 0; n < k_fftInputLen; n++)
    {
        windowedDataI[n].real = filterVec[n] * inputArrayVec[n + frameOffset];
        windowedDataI[n].imag = 0.0;
    }
}
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
__device__ void FFTkernelRecursiveCVersion(Comp* windowedDataI, int inputSize)
{
    if (inputSize <= 1)
    {
        return;
    }

    const float PI = (float)3.141592653589793238460;
    int start, size, stride;
    float polarMagnitude;
    float theta;

    //replacement for slice
    size = inputSize / 2;
    stride = 2;

    Comp* evenSlice = (Comp*)malloc(size*sizeof(Comp));
    Comp* oddSlice = (Comp*)malloc(size*sizeof(Comp));

    //divide
    for (int i = 0; i < size; i++)
    {
        evenSlice[i] = windowedDataI[(i * stride)];
        oddSlice[i] = windowedDataI[1 + (i * stride)];
    }
    
    //conquer
    FFTkernelRecursiveCVersion(evenSlice, size);
    FFTkernelRecursiveCVersion(oddSlice, size);
    

    
    for (size_t k = 0; k < size / 2; ++k)
    {
        
        Comp t;
        float theta = -2 * PI * k / size; //radians
        //confirmed replaces std::polar below
        t.real = (float)(cosf(theta) * oddSlice[k].real);
        t.real -= (float)(sinf(theta) * oddSlice[k].imag);

        t.imag = (float)(cosf(theta) * oddSlice[k].imag);
        t.imag += (float)(sinf(theta) * oddSlice[k].real);
        
        //x[k] = even[k] + t
        windowedDataI[k].real = evenSlice[k].real + t.real;
        windowedDataI[k].imag = evenSlice[k].imag + t.imag;

        //x[k + N / 2] = even[k] - t;
        windowedDataI[k + (size / 2)].real = evenSlice[k].real - t.real;
        windowedDataI[k + (size / 2)].imag = evenSlice[k].imag - t.imag;
    }

    free(evenSlice);
    free(oddSlice);
}
  
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

//todo this needs to receive pointer to return memory
__global__ void FFTkernel(float* filterVec, int* frameOffsetsVec, float* inputArrayVec, Comp* windowedData)
{
    int i = threadIdx.x;

    int windowedDataOffset = i * k_fftInputLen;
    Comp* windowedDataI = windowedData + windowedDataOffset; 

    kernelWindowData(i, filterVec, inputArrayVec, windowedDataI);

# if __CUDA_ARCH__>=200
    printf("start of FFT calc\n");
#endif

    FFTkernelRecursiveCVersion(windowedDataI, k_fftInputLen);

# if __CUDA_ARCH__>=200
    for (int i = 0; i < k_fftInputLen / 2; i++)
    {
        printf("%f \n", windowedDataI[i].imag);
    }
    printf("End of FFT calc\n");
#endif

    # if __CUDA_ARCH__>=200
    //printf("%d \n", frameOffsetsVec[i]);
    #endif
}

 /**********************************************************
  * Main
  **********************************************************/
int main()
{
    std::cout << "Start of program\n";

    //Read CSV file and put elements in inputArray vector
    ContiguousArray<float> inputArray = readCSV();

    //generate blackman-harris filter from 0 to k_fftInputLen-1 to window the input data
    ContiguousArray<float> filter = generateFilter();

    //generate frameOffsets
    ContiguousArray<int> frameOffsets = generateFrameOffsets();//list of frame offsets used by workers

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

cudaError_t FFTWithCuda(ContiguousArray<float> filter, ContiguousArray<int> frameOffsets, ContiguousArray<float> inputArray)
{
    cudaError_t cudaStatus;
    float* dev_filter = 0;
    int* dev_frameOffsets = 0;
    float* dev_inputArray = 0;
    Comp* dev_windowedData = 0;

    //todo debug, only run one thread until that case works
    const int const threadCount = 1;///////////////frameOffsets.numElements;

    // Choose which GPU to run on, change this on a multi-GPU system
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    cudaStatus = cudaThreadSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaThreadSynchronize failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    //Allocate space on GPU for the filter, frameOffsets, and inputArray.  Only one of each is needed

    cudaStatus = cudaMalloc((void**)&dev_filter, filter.getSize());
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_frameOffsets, frameOffsets.getSize());
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_inputArray, inputArray.getSize());
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    //Allocate room on GPU for windowed data (windowing is run in parallel), data initialized on GPU so no memcpy for this data
    cudaStatus = cudaMalloc((void**)&dev_windowedData, (k_fftInputLen*threadCount*sizeof(Comp)));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    //Copy input vectors from host memory to GPU buffers
    cudaStatus = cudaMemcpy(dev_filter, filter.ptr, filter.getSize(), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_frameOffsets, frameOffsets.ptr, frameOffsets.getSize(), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_inputArray, inputArray.ptr, inputArray.getSize(), cudaMemcpyHostToDevice);//todo replace global
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread per frameOffset

    FFTkernel << <1, threadCount >> > (dev_filter, dev_frameOffsets, dev_inputArray, dev_windowedData);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "FFTKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching FFTKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    ContiguousArray<float> outputData;
    outputData.numElements = k_fftInputLen * threadCount;
    outputData.ptr = new float[outputData.numElements];

    //TODO this memcpy fails
    //cudaStatus = cudaMemcpy(outputData.ptr, dev_windowedData, outputData.getSize(), cudaMemcpyDeviceToHost);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMemcpy failed!");
    //    goto Error;
    //}

Error:
    //Free input data
    cudaFree(dev_filter);
    cudaFree(dev_frameOffsets);
    cudaFree(dev_inputArray);

    //Free output data
    cudaFree(dev_windowedData);

    return cudaStatus;
}