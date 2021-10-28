/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512


__global__ void naiveReduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    // NAIVE REDUCTION IMPLEMENTATION
    
    __shared__ float sum[2*BLOCK_SIZE];
    unsigned int a=threadIdx.x;
    unsigned int i=2*blockIdx.x*blockDim.x;
    
    sum[a]=in[i+a];
    sum[blockDim.x+a]=in[i+blockDim.x+a];
    for(unsigned int j=1;j<=blockDim.x;j*=2)
    {
        __syncthreads();
        if(a%j==0)
            sum[2*a]+=sum[2*a+j];
    }
    if(a==0)
        out[blockIdx.x]=sum[0];

}

__global__ void optimizedReduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    // OPTIMIZED REDUCTION IMPLEMENTATION
    
    __shared__ float sum[2*BLOCK_SIZE];
    unsigned int a=threadIdx.x;
    unsigned int i=2*blockIdx.x*blockDim.x;
    
    if(a+i<size)
        sum[a]=in[a+i];
    else
        sum[a]=0.0;
    
    if(blockDim.x+a+i<size)
        sum[blockDim.x+a]=in[blockDim.x+a+i];
    else
        sum[blockDim.x+a]=0.0;
    
    for(unsigned int j=blockDim.x;j>0;j/=2)
    {
         __syncthreads();
        if(a<j)
            sum[a]+=sum[a+j];
    }
    if(a==0)
        out[blockIdx.x]=sum[0];


}

