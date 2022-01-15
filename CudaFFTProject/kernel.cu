
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

//Function declarations
ContiguousArray<double> readCSV();
ContiguousArray<int> generateFrameOffsets();
ContiguousArray<double> generateFilter();
std::vector<double> windowData(int frameOffset, std::vector<double> filter);
void fft(CArray& x);
cudaError_t FFTWithCuda(ContiguousArray<double> filter, ContiguousArray<int> frameOffsets, ContiguousArray<double> inputArray);

//Global variables
int inputArraySize = 0;

//Global FFT variables
const double PI = 3.141592653589793238460;
typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;

//user set parameters
const int k_fftInputLen = 100; //length of FFT input array(data points per FFT frame)
const int k_fftFrameOffset = 10; //offset between start of FFT frames(eg x[n]=x[n-1]+k_fftFrameOffset where x[n] is the first value used as input to the fft frame)

/**********************************************************
 * Functions run on single thread
 **********************************************************/
ContiguousArray<double> readCSV()
{
    std::vector<double> csvVector;
    ContiguousArray<double> retArray;

    const char delimeter = ',';//delimeter between items in CSV file
    std::string line;
    std::string string;

    std::ifstream myFile("data.CSV");

    if (!myFile.is_open()) throw std::runtime_error("Couldn't open file");

    while (getline(myFile, string, delimeter)) {
        csvVector.push_back(std::stod(string));
        inputArraySize++;
    }

    //CUDA requires contiguous memory
    retArray.numElements = inputArraySize;
    retArray.ptr = new double[retArray.numElements];

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

ContiguousArray<double> generateFilter()
{
    //w[n] = a0 - a1*cos(x) + a2*cos(2x) - a3cos(3x), x = (2n*pi)/N, 0 < n < N
    const double a0 = 0.35875;
    const double a1 = 0.48829;
    const double a2 = 0.14128;
    const double a3 = 0.01168;

    ContiguousArray<double> retArray;
    double x;
    double term1, term2, term3;
    double w_n;

    retArray.numElements = k_fftInputLen;
    retArray.ptr = new double[retArray.numElements];

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

__device__ void kernelWindowData(int frameOffset, double* filterVec, double* inputArrayVec, double* windowedDataI)
{
    for (int n = 0; n < k_fftInputLen; n++)
    {
        windowedDataI[n] = filterVec[n] * inputArrayVec[n + frameOffset];
    }
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//// Cooley–Tukey FFT (in-place)
//void fft(CArray& x)
//{
//    const size_t N = x.size();
//    if (N <= 1) return;
//
//    // divide
//    CArray even = x[std::slice(0, N / 2, 2)];
//    CArray  odd = x[std::slice(1, N / 2, 2)];
//
//    // conquer
//    fft(even);
//    fft(odd);
//
//    // combine
//    for (size_t k = 0; k < N / 2; ++k)
//    {
//        Complex t = std::polar(1.0, -2 * PI * k / N) * odd[k];
//        x[k] = even[k] + t;         //todo this is overwriting the input data, need to preserve input copy
//        x[k + N / 2] = even[k] - t; //todo this is overwriting the input data, need to preserve input copy
//    }
//}
 
__device__ void FFTkernelRecursiveCVersion(double* windowedDataI, int inputSize)
{
    if (inputSize <= 1)
    {
        return;
    }

    const double PI = 3.141592653589793238460;
    int start, size, stride;
    double polarMagnitude;
    double theta;

    //replacement for slice
    size = inputSize / 2;
    stride = 2;
    
    double* evenSlice = (double*)malloc(size);
    double* oddSlice = (double*)malloc(size);

    //divide
    for (int i = 0; i < size; i++)
    {
        evenSlice[i] = windowedDataI[(i * stride)];
        oddSlice[i] = windowedDataI[1 + (i * stride)];
    }
    
    //conquer
    FFTkernelRecursiveCVersion(evenSlice, size);
    FFTkernelRecursiveCVersion(oddSlice, size);

    double* x = (double*)malloc(size);
    
    for (size_t k = 0; k < size / 2; ++k)
    {
        
        double oddIndex = oddSlice[k];
        double evenIndex = evenSlice[k];

        double theta = -2 * PI * k / size; //radians
        theta *= 57.29578; //to degrees
        //confirmed replaces std::polar below
        double t_real = cos(theta) * oddIndex;
        double t_imag = sin(theta) * oddIndex;

        //works up to here
        
        //x[k] = evenIndex + t;
        //Calculate x[k]
        double xk_real = evenIndex + t_real;
        xk_real *= xk_real;
        double xk_imag = t_imag;
        xk_imag *= xk_imag;
        x[k] = sqrt(xk_real + xk_imag);
        /*todo this chunk breaks on cuda, probably sqrt of an invalid value
        * 
        //Calculate x[k + size / 2]
        //todo this isn't working.... makes a very large value
        double xk2_real = evenIndex - t_real;
        xk2_real *= xk2_real;
        double xk2_imag = -t_imag;
        xk2_imag *= xk2_imag;
        x[k + size / 2] = sqrt(xk2_real + xk2_imag);
        */
    }
    //todo free(evenSlice) and oddslice here
# if __CUDA_ARCH__>=200
    //printf("%f \n", x[j]);
    //printf("%f \n", windowedDataI[j]);
    //printf("%f \n", polarMagnitude);
#endif
}
  
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

//todo this needs to receive pointer to return memory
__global__ void FFTkernel(double* filterVec, int* frameOffsetsVec, double* inputArrayVec, double* windowedData)
{
    int i = threadIdx.x;

    int windowedDataOffset = i * k_fftInputLen;
    double* windowedDataI = windowedData + windowedDataOffset; 

    kernelWindowData(i, filterVec, inputArrayVec, windowedDataI);

# if __CUDA_ARCH__>=200
    printf("start of FFT calc\n");
#endif

    FFTkernelRecursiveCVersion(windowedDataI, k_fftInputLen);

# if __CUDA_ARCH__>=200
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
    ContiguousArray<double> inputArray = readCSV();

    //generate blackman-harris filter from 0 to k_fftInputLen-1 to window the input data
    ContiguousArray<double> filter = generateFilter();

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

cudaError_t FFTWithCuda(ContiguousArray<double> filter, ContiguousArray<int> frameOffsets, ContiguousArray<double> inputArray)
{
    cudaError_t cudaStatus;
    double* dev_filter = 0;
    int* dev_frameOffsets = 0;
    double* dev_inputArray = 0;
    double* dev_windowedData = 0;

    //todo debug, only run one thread until that case works
    const int const threadCount = 1;///////////////frameOffsets.numElements;

    // Choose which GPU to run on, change this on a multi-GPU system
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
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
    cudaStatus = cudaMalloc((void**)&dev_windowedData, (k_fftInputLen*threadCount));
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
    ContiguousArray<double> outputData;
    outputData.numElements = k_fftInputLen * threadCount;
    outputData.ptr = new double[outputData.numElements];

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