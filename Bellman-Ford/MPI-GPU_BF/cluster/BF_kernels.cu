

/**********************************************************************************

Copyright 2021 Charles W. Johnson
 
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
  
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 
**********************************************************************************/


//
// Filename: BF_kernels.cu
// Author: Charles W Johnson
// Description: Kernels for MPI GPU-based Bellman-Ford algorithm
//


#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "BF_kernels.cuh"

using namespace std;


/* ---- CPU/Wrapper Functions ---- */

// Name: initialize_dp_array_Wrapper
//
// Description: CPU wrapper function to execute the kernel to initialize the dp array
//
//
void initialize_dp_array_Wrapper(dim3 grid, dim3 blocks, distPred* d_dp, int infinity, int source, int num_vertices)
{
    cudaGetLastError(); // reset the runtime error variable to cudaSuccess

    initialize_dp_array<<<grid, blocks>>>(d_dp, infinity, source, num_vertices);
}


// Name: BellmanFord_GPU_Wrapper
//
// Description: CPU wrapper function to execute the B-F kernel
//
//
void BellmanFord_GPU_Wrapper(int proc_start_edge, int proc_num_edges, uint32_t num_vertices, uint32_t num_edges,
                             Edge* d_edgeList, distPred* d_dp, bool BF_short, bool& finished,
                             dim3 grid, dim3 blocks)
{
    // since CUDA is whining about using bools and I can't find a fix,
    // I'm going to use ints - 1 is true, 0 is false

    int h_change = 0;
    int *d_change = 0;
    cudaMalloc((void**) &d_change, sizeof(int));

    finished = true;

    for (int i=1; i < (num_vertices-1); i++)
    {
        // we make the local change false
        h_change = 0;

        // we copy the local value to the device
        cudaMemcpy(d_change, &h_change, sizeof(int), cudaMemcpyHostToDevice);

        // we then run the kernel
        cudaBellmanFord<<<grid, blocks>>>(proc_start_edge, proc_num_edges, d_edgeList, 
                                          d_dp, d_change);

        cudaDeviceSynchronize();
    
        // we now copy the value from the device back to the local variable
        cudaMemcpy(&h_change, d_change, sizeof(int), cudaMemcpyDeviceToHost);

        // if the device is reporting a change, then we are not finished
        //
        if (h_change == 1) {
            finished = false;
        }

        if (BF_short == true) {
            if (!h_change) {
                break;
            }
        }
    }

    cudaFree(d_change);
}


/* ---- Kernels ---- */

// Name: initialize_dp_array
//
// Description: Initializes the d_dp array
//
//
__global__ void initialize_dp_array(distPred* d_dp, int infinity, int source, int num_vertices)
{
    uint32_t tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (tid < num_vertices)
    {
        d_dp[tid].dist = infinity;
        d_dp[tid].pred = (int)NULL;
    
        if (tid == source) {
            d_dp[tid].dist = 0;
        }
    }
}


// Name: cudaBellmanFord
//
// Description: Executes the Bellman-Ford algorithm on the specified edges
//
//
__global__ void cudaBellmanFord(int proc_start_edge, int proc_num_edges, Edge* d_edgeList, 
                                distPred* d_dp, int* d_change)
{
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    int edge = proc_start_edge + tid;

    int u, v, w;

    if (edge < (proc_start_edge + proc_num_edges))
    {
        u = d_edgeList[edge].u;
        v = d_edgeList[edge].v;
        w = d_edgeList[edge].w;

        if ((d_dp[u].dist + w) < d_dp[v].dist)
        {
            d_dp[v].dist = (d_dp[u].dist + w);
            d_dp[v].pred = u;

            *d_change = 1;
        }
    }
}


