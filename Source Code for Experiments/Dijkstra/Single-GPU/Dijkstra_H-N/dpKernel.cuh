

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
// Filename: dpKernel.cuh
// Author: Charles W Johnson
// Description: Header file for dp kernel for Dijkstra's algorithm
//

#ifndef DP_KERNEL_H_
#define DP_KERNEL_H_


#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Dijkstra_custom_data_structures.h"


/* ---- Wrapper Functions ---- */

// Name: initialize_dp_array_wrapper
//
// Description: Wrapper function for initialize_dp_array kernel
//
//
void initialize_dp_array_wrapper(distPred* d_dp, distPred* d_dp_updating_cost, int max_cost,
                                    int source, int num_vertices, bool* d_graph_mask, dim3 grid, dim3 blocks);


/* ---- Kernels ---- */

// Name: initialize_dp_array
//
// Description: Initializes the dp arrays and the graph mask
//
//
__global__ void initialize_dp_array(distPred* d_dp, distPred* d_dp_updating_cost, int max_cost,
                                    int source, int num_vertices, bool* d_graph_mask);


#endif /* DP_KERNEL_H_ */
