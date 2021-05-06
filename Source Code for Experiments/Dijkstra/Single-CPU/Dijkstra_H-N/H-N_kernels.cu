

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
 

-------


This additional MIT license was added so as to cover my edits to Harish & Narayanan's
code. In terms of the implementation (the kernels and the control loop, etc), the
changes have been limited to renaming variables, adding necessary wrapper functions,
and then adding support so that the program can keep track of the predecessors,
hence their license remaining on the code. My license is added to cover my edits and
additions, most of which were to allow the program to operate in the framework of
my thesis programs (common file format, etc).

Charles W. Johnson
May, 2021

**********************************************************************************/


/*************************************************************************************
Implementing Single Source Shortest Path on CUDA 1.1 Hardware using algorithm 
given in HiPC'07 paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
All rights reserved.
  
Permission to use, copy, modify and distribute this software and its documentation for 
educational purpose is hereby granted without fee, provided that the above copyright 
notice and this permission notice appear in all copies of this software and that you do 
not sell the software.
  
THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
OTHERWISE.

This Kernel updates the cost of each neighbour using atomicMin operation on CUDA 1.1 
hardware. Note that this operation is not supported on CUDA 1.0 hardware.

Created by Pawan Harish.
**************************************************************************************/


//
// Filename: H-N_kernels.cu
// Author: Charles W Johnson
// Description: Algorithm kernels for Dijkstra's algorithm
//


#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "H-N_kernels.cuh"


/* ---- Wrapper Functions ---- */

// Name: DijkstraKernel1_wrapper
//
// Description: Wrapper function for DijkstraKernel1
//
//
void DijkstraKernel1_wrapper(int* d_V, int* d_E, short int* d_W, 
                             bool* g_graph_mask, 
                             int num_vertices, int num_edges,
                             distPred* d_dp, distPred* d_dp_updating_cost,
                             dim3 grid, dim3 blocks)
{

    DijkstraKernel1<<<grid, blocks>>>(d_V, d_E, d_W, 
                                      g_graph_mask, 
                                      num_vertices, num_edges,
                                      d_dp, d_dp_updating_cost);
    cudaDeviceSynchronize();
}


// Name: DijkstraKernel2_wrapper
//
// Description: Wrapper function for DijkstraKernel2
//
//
void DijkstraKernel2_wrapper(int* d_V, int* d_E, short int* d_W,
                             bool* g_graph_mask, 
                             bool *d_finished, int num_vertices, int num_edges,
                             distPred* d_dp, distPred* d_dp_updating_cost,
                             dim3 grid, dim3 blocks)
{

    DijkstraKernel2<<<grid, blocks>>>(d_V, d_E, d_W,
                                      g_graph_mask, 
                                      d_finished, num_vertices, num_edges,
                                      d_dp, d_dp_updating_cost);
    cudaDeviceSynchronize();
}


/* ---- Kernels ---- */

// Name: DijkstraKernel1
//
// Description: Relaxes the edges and calculates and stores (PRN) update costs
//
//
__global__ void DijkstraKernel1(int* d_V, int* d_E, short int* d_W, 
                                bool* g_graph_mask, 
                                int num_vertices, int num_edges,
                                distPred* d_dp, distPred* d_dp_updating_cost)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int i, id;
    int end = num_edges;

    int old_value = 0;

    if ((tid < num_vertices) && g_graph_mask[tid])
    {
        if (tid < (num_vertices-1)) {
            end = d_V[tid+1];
        }

        for (i = d_V[tid]; i<end; i++)
        {
            id = d_E[i];

            old_value = atomicMin(&d_dp_updating_cost[id].dist, (d_dp[tid].dist + d_W[i]));

            if ((d_dp[tid].dist + d_W[i]) < old_value)
            {
                d_dp_updating_cost[id].pred = tid;
            } 
        }

        g_graph_mask[tid]=false;
    }
}


// Name: DijkstraKernel2
//
// Description: Updates the costs PRN
//
//
__global__ void DijkstraKernel2(int* d_V, int* d_E, short int* d_W, 
                                bool* g_graph_mask, 
                                bool *d_finished, int num_vertices, int num_edges,
                                distPred* d_dp, distPred* d_dp_updating_cost)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ((tid < num_vertices) && (d_dp[tid].dist > d_dp_updating_cost[tid].dist))
    {
        d_dp[tid] = d_dp_updating_cost[tid];
        g_graph_mask[tid] = true;
        *d_finished = true;
    }

    if (tid<num_vertices) {
        d_dp_updating_cost[tid] = d_dp[tid];
    }
}


