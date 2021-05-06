

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
// Filename: H-N_kernels.cuh
// Author: Charles W Johnson
// Description: Header file for algorithm kernels for Dijkstra's algorithm
//


#ifndef HN_KERNELS_H_
#define HN_KERNELS_H_

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Dijkstra_custom_data_structures.h"


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
                             dim3 grid, dim3 blocks);


// Name: DijkstraKernel2_wrapper
//
// Description: Wrapper function for DijkstraKernel2
//
//
void DijkstraKernel2_wrapper(int* d_V, int* d_E, short int* d_W, 
                             bool* g_graph_mask, 
                             bool *d_finished, int num_vertices, int num_edges,
                             distPred* d_dp, distPred* d_dp_updating_cost,
                             dim3 grid, dim3 blocks);


/* ---- Kernels ---- */

// Name: DijkstraKernel1
//
// Description: Relaxes the edges and calculates and stores (PRN) update costs
//
//
__global__ void DijkstraKernel1(int* d_V, int* d_E, short int* d_W, 
                                bool* g_graph_mask, 
                                int num_vertices, int num_edges,
                                distPred* d_dp, distPred* d_dp_updating_cost);


// Name: DijkstraKernel2
//
// Description: Updates the costs PRN
//
//
__global__ void DijkstraKernel2(int* d_V, int* d_E, short int* d_W, 
                                bool* g_graph_mask, 
                                bool *d_finished, int num_vertices, int num_edges,
                                distPred* d_dp, distPred* d_dp_updating_cost);


#endif /* HN_KERNELS_H_ */
