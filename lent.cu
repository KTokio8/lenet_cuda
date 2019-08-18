#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "header.h"

#define IMAGE_FILE       "./txt/image1000/"
#define CHECK_PARAMS    (0)

#define IMAGE_SIZE      (1 * 28 * 28)

#define CONV1_W_SIZE    (20 * 1 * 5 * 5)
#define CONV1_B_SIZE    (20)
#define CONV1_OUT_SIZE  (20 * 24 * 24)

#define POOL1_OUT_SIZE  (20 * 12 * 12)

#define CONV2_W_SIZE    (50 * 20 * 5 * 5)
#define CONV2_B_SIZE    (50)
#define CONV2_OUT_SIZE  (50 * 8 * 8)

#define POOL2_OUT_SIZE  (50 * 4 * 4)

#define FC1_W_SIZE      (500 * 800)
#define FC1_B_SIZE      (500)
#define FC1_OUT_SIZE    (500)

#define FC2_W_SIZE      (10 * 500)
#define FC2_B_SIZE      (10)
#define FC2_OUT_SIZE    (10)

#define CUDA_SAFE_CALL(func)                                                \
    do {                                                                    \
        cudaError_t err = (func);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n",  \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);      \
            exit(err);                                                      \
        }                                                                   \
    } while (0)

//Utils
void* deviceMalloc(size_t size) {
    void* ret;
    CUDA_SAFE_CALL(
        cudaMalloc(&ret, size));
    return ret;
}

void* sendToHost(void* src, size_t size) {
    void* dst = deviceMalloc(size);
    CUDA_SAFE_CALL(
        cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    return dst;
}

void* sendToDevice(void* src, size_t size) {
    void* dst = malloc(size);
    CUDA_SAFE_CALL(
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    return dst;
}


/*Convolution*/
template <int insize,  int InputChannel,  int insize_2, 
          int outsize, int outsize_2,
          int kernel, int kernel2>
__global__ void convolution_gpu(float* inputImage, float* outImage, 
                       float* weight, float* bias)
{
    __shared__ float sharedImage[insize_2];

    int thread_x = threadIdx.x; 
    int thread_y = threadIdx.y; 
    int block_x = blockIdx.x;
    int pos = thread_x + insize * thread_y; 
    int diff = (insize - outsize) >> 1;
    bool outOfRange = thread_x > outsize+1 || thread_x < diff || thread_y > outsize+1 || thread_y < diff;

    float sum = 0;
  
    for ( int ch = 0; ch < InputChannel; ch++) {
        __syncthreads();
        sharedImage[pos] = inputImage[pos + insize_2 * ch]; 

        __syncthreads();

        if (outOfRange) {
            continue;
        }

         int kchan = kernel2 * ch + kernel2 * InputChannel * block_x;

        
        for ( int i = 0; i < kernel; i++) {
          
            for ( int j = 0; j < kernel; j++) {
                 int kPos = j + kernel * i + kchan;

                sum += sharedImage[thread_x-diff+j + insize * (thread_y-diff+i)] * weight[kPos];
            }
        }
    }

    if (outOfRange) {
        return;
    }

    outImage[(thread_x-diff) + (thread_y-diff) * outsize + block_x * outsize_2] = sum + bias[block_x];

}


/*Maxpooling_gpu*/
template<int insize, int insize_2, int outsize, int outsize_2> 
__global__ void maxpooling_gpu(float* inputImage, float* outImage)
{
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    int block_x = blockIdx.x;
    int thread_x2 = thread_x * 2;
    int thread_y2 = thread_y * 2;
    int ch = block_x * insize_2;


    outImage[thread_x + outsize * thread_y + block_x * outsize_2] = fmaxf(
        fmaxf(inputImage[thread_x2 + insize *  thread_y2    + ch], inputImage[(thread_x2+1) + insize *  thread_y2    + ch]),
        fmaxf(inputImage[thread_x2 + insize * (thread_y2+1) + ch], inputImage[(thread_x2+1) + insize * (thread_y2+1) + ch])
    ); 
}


/*Fc*/
template <int insize, int outsize>
__global__ void Fc_softmax_gpu(float* input, float* output, float* weight, float* bias)
{

    int thread_x = threadIdx.x;
    int block_x = blockIdx.x;

    __shared__ float sharedOut[insize];

    sharedOut[thread_x] = input[thread_x] * weight[thread_x + insize * block_x]; 
    __syncthreads();

     int j = 0;
    for ( int i = insize >> 1; i > 0;i >>= 1){
        if (thread_x < i) {
            sharedOut[thread_x] += sharedOut[thread_x + i];
            if (j == 1 && thread_x == 0) {
                sharedOut[thread_x] += sharedOut[i << 1];
            }
        }

        __syncthreads();

        j = i & 1;
    }
    
    if (thread_x == 0){
        output[block_x] = expf(sharedOut[0] + bias[block_x]);
    }
}

/* Relu*/
template <int insize, int outsize>
__global__ void Fc_relu_gpu(float* input, float* output, float* weight, float* bias)
{

    int thread_x = threadIdx.x;
    int block_x = blockIdx.x;
    __shared__ float sharedOut[insize];


    sharedOut[thread_x] = input[thread_x] * weight[thread_x + insize * block_x]; 

    __syncthreads();
     int j = 0;
    for ( int i = insize >> 1; i > 0; i >>= 1 ){

        if (thread_x < i) {
            sharedOut[thread_x] += sharedOut[thread_x + i];

            if (j == 1 && thread_x == 0) {

                sharedOut[thread_x] += sharedOut[i << 1];
            }
        }
        __syncthreads();
       j = i & 1;
    }
    
    if (thread_x == 0){
        output[block_x] = fmaxf(0, sharedOut[0] + bias[block_x]);
    }
}


int main()
{
    char imagefile[64];
    //char s[32];

    /* allocate host variables */
    float* hostImage = (float*) malloc(sizeof(float) * IMAGE_SIZE);

    float* hostConv1W = (float*) malloc(sizeof(float) * CONV1_W_SIZE);
    float* hostConv1B = (float*) malloc(sizeof(float) * CONV1_B_SIZE);
    float* hostConv1O = (float*) malloc(sizeof(float) * CONV1_OUT_SIZE);

    float* hostPool1O = (float*) malloc(sizeof(float) * POOL1_OUT_SIZE);

    float* hostConv2W = (float*) malloc(sizeof(float) * CONV2_W_SIZE);
    float* hostConv2B = (float*) malloc(sizeof(float) * CONV2_B_SIZE);
    float* hostConv2O = (float*) malloc(sizeof(float) * CONV2_OUT_SIZE);

    float* hostPool2O = (float*) malloc(sizeof(float) * POOL2_OUT_SIZE);

    float* hostFc1W = (float*) malloc(sizeof(float) * FC1_W_SIZE);
    float* hostFc1B = (float*) malloc(sizeof(float) * FC1_B_SIZE);
    float* hostFc1O = (float*) malloc(sizeof(float) * FC1_OUT_SIZE);

    float* hostFc2W = (float*) malloc(sizeof(float) * FC2_W_SIZE);
    float* hostFc2B = (float*) malloc(sizeof(float) * FC2_B_SIZE);

    float* hostFc2O = (float*) malloc(sizeof(float) * FC2_OUT_SIZE);
    float* gpuFc2O = (float*) malloc(sizeof(float) * FC2_OUT_SIZE);

    /* Rread prameters*/ 
    print_params("IMAGE", hostImage, IMAGE_SIZE);

    read_params("./txt/conv1_w.txt", hostConv1W, CONV1_W_SIZE);
    print_params("CONV1_W", hostConv1W, CONV1_W_SIZE);
    read_params("./txt/conv1_b.txt", hostConv1B, CONV1_B_SIZE);
    print_params("CONV1_B", hostConv1B, CONV1_B_SIZE);

    read_params("./txt/conv2_w.txt", hostConv2W, CONV2_W_SIZE);
    print_params("CONV2_W", hostConv2W, CONV2_W_SIZE);
    read_params("./txt/conv2_b.txt", hostConv2B, CONV2_B_SIZE);
    print_params("CONV2_B", hostConv2B, CONV2_B_SIZE);

    read_params("./txt/fc1_w.txt", hostFc1W, FC1_W_SIZE);
    print_params("FC1_W", hostFc1W, FC1_W_SIZE);
    read_params("./txt/fc1_b.txt", hostFc1B, FC1_B_SIZE);
    print_params("FC1_B", hostFc1B, FC1_B_SIZE);

    read_params("./txt/fc2_w.txt", hostFc2W, FC2_W_SIZE);
    print_params("FC2_W", hostFc2W, FC2_W_SIZE);
    read_params("./txt/fc2_b.txt", hostFc2B, FC2_B_SIZE);
    print_params("FC2_B", hostFc2B, FC2_B_SIZE);
    printf("\n");
 
    /* allocate device variables */
    float* devImage = (float*) sendToHost(hostImage, sizeof(float) * IMAGE_SIZE);

    float* devConv1W = (float*) sendToHost(hostConv1W, sizeof(float) * CONV1_W_SIZE);
    float* devConv1B = (float*) sendToHost(hostConv1B, sizeof(float) * CONV1_B_SIZE);
    float* devConv1O = (float*) deviceMalloc(sizeof(float) * CONV1_OUT_SIZE);
    float* devPool1O = (float*) deviceMalloc(sizeof(float) * POOL1_OUT_SIZE);

    float* devConv2W = (float*) sendToHost(hostConv2W, sizeof(float) * CONV2_W_SIZE);
    float* devConv2B = (float*) sendToHost(hostConv2B, sizeof(float) * CONV2_B_SIZE);
    float* devConv2O = (float*) deviceMalloc(sizeof(float) * CONV2_OUT_SIZE);
    float* devPool2O = (float*) deviceMalloc(sizeof(float) * POOL2_OUT_SIZE);

    float* devFc1W = (float*) sendToHost(hostFc1W, sizeof(float) * FC1_W_SIZE);
    float* devFc1B = (float*) sendToHost(hostFc1B, sizeof(float) * FC1_B_SIZE);
    float* devFc1O = (float*) deviceMalloc(sizeof(float) * FC1_OUT_SIZE);

    float* devFc2W = (float*) sendToHost(hostFc2W, sizeof(float) * FC2_W_SIZE);
    float* devFc2B = (float*) sendToHost(hostFc2B, sizeof(float) * FC2_B_SIZE);
    float* devFc2O = (float*) deviceMalloc(sizeof(float) * FC2_OUT_SIZE);

    dim3 conv1Grid(20, 1, 1);
    dim3 conv1Block(28, 28, 1);

    dim3 pool1Grid(20, 1, 1);
    dim3 pool1Block(12, 12, 1);

    dim3 conv2Grid(50, 1, 1);
    dim3 conv2Block(12, 12, 1);

    dim3 pool2Grid(50, 1, 1);
    dim3 pool2Block(4, 4, 1);

    dim3 Fc1Grid(500, 1, 1);
    dim3 Fc1Block(800, 1, 1);
    
    dim3 Fc2Grid(10, 1, 1);
    dim3 Fc2Block(500, 1, 1);

    printf("\n");

    printf("/// LeNet ///\n");
    fflush(stdout);
  
    printf("Memory allocation ...\n");
    fflush(stdout);

    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsedTime;
    float cpuTime;
    float gpuTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

   
    int cnt;
    for(cnt = 0; cnt < 1000; cnt++) {
        sprintf(imagefile, "%simage%03d.txt", IMAGE_FILE,cnt); 
        printf("%s\n", imagefile);
        fflush(stdout);

        read_params(imagefile, hostImage, IMAGE_SIZE);
        norm_image(hostImage, IMAGE_SIZE);

        //show_image(hostImage, 28);
        //printf("\n");


        /* Feed forward on CPU */
        printf("feed forward on cpu... \n");
        fflush(stdout);

        cudaEventRecord(start, 0);

        
        convolution(hostImage, 28, 1, hostConv1O, 24, 20, hostConv1W, hostConv1B, 5, 1);
        maxpooling(hostConv1O, 24, 20, hostPool1O, 12, 2, 2);
        convolution(hostPool1O, 12, 20, hostConv2O, 8, 50, hostConv2W, hostConv2B, 5, 1);
        maxpooling(hostConv2O, 8, 50, hostPool2O, 4, 2, 2);

        classifier(hostPool2O, 800, hostFc1O, 500, hostFc1W, hostFc1B);
        relu(hostFc1O, 1, 500);
        classifier(hostFc1O, 500, hostFc2O, 10, hostFc2W, hostFc2B);
        softmax(hostFc2O, 10);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cpuTime += elapsedTime;

        printf("CPU time: %f ms \n", elapsedTime);

        /* Feed forward on GPU */
        printf("feed forword on gpu");
        CUDA_SAFE_CALL(
            cudaMemcpy(devImage, hostImage, IMAGE_SIZE * sizeof(float),
            cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(
            cudaMemcpy(hostImage, devImage, IMAGE_SIZE * sizeof(float),
            cudaMemcpyDeviceToHost));

        //show_image(hostImage, 28);

        cudaEventRecord(start, 0);

        convolution_gpu<28,1,784,24,576,5,25><<<conv1Grid, conv1Block>>>(
            devImage, devConv1O, devConv1W, devConv1B);
        maxpooling_gpu<24,576,12,144><<<pool1Grid, pool1Block>>>(
            devConv1O, devPool1O);
        convolution_gpu<12,20,144,8,64,5,25><<<conv2Grid, conv2Block>>>(
            devPool1O, devConv2O, devConv2W, devConv2B);
        maxpooling_gpu<8,64,4,16><<<pool2Grid, pool2Block>>>(
            devConv2O, devPool2O);

        Fc_relu_gpu<800,500><<<Fc1Grid, Fc1Block>>>(
            devPool2O, devFc1O, devFc1W, devFc1B);
        Fc_softmax_gpu<500,10><<<Fc2Grid, Fc2Block>>>(
            devFc1O, devFc2O, devFc2W, devFc2B);


        CUDA_SAFE_CALL(
            cudaMemcpy(gpuFc2O, devFc2O,
                        FC2_OUT_SIZE * sizeof(float),
                        cudaMemcpyDeviceToHost));

        int i,j;
        float sum = 0;
        for (i = 0; i < FC2_OUT_SIZE; i++) {
            sum += gpuFc2O[i];
        }
        for(j = 0; j < FC2_OUT_SIZE; j++) {
            gpuFc2O[j] /= sum;
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);

        gpuTime += elapsedTime;

        
        print_all_params(hostFc2O, 10);
        

        printf("GPU time: %f ms \n", elapsedTime);
        print_all_params(gpuFc2O, 10);
        printf("\n");

    }

    /* Reset device */
    CUDA_SAFE_CALL(cudaDeviceReset());

    printf("AverageTime \n CPU: %f ms,\n GPU: %f ms \n", cpuTime/1000.0f,  gpuTime/ 1000.0f);
    printf("Ratio of GPU to CPU: %f \n", cpuTime/gpuTime);

    return EXIT_SUCCESS;
}




