

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
// Filename: Kernels.cuh
// Author: Charles W Johnson
// Description: Header file for wrapper functions and kernels for Dijkstra's algorithm
//
// Note: While the Dijkstra kernel below is your basic Dijkstra kernel,
//       due to the fact that it is also closely based on the Dijkstra
//       implementation by Harish & Narayanan (the simplified version
//       of their implementation is pretty much the same as the basic
//       Dijkstra kernel), then they deserve partial credit for the
//       code below.
//

#ifndef KERNELS_H_
#define KERNELS_H_


#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Dijkstra_custom_data_structures.h"


/* ---- Wrapper functions ---- */

// Name: initialize_GPU_arrays_wrapper
//
// Description: Wrapper function that initializes the dp and vertex_settled arrays
//
//
void initialize_GPU_arrays_wrapper(distPred* d_dp, bool* d_vertex_settled, int infinity, int source,
                                  int num_vertices, dim3 grid, dim3 blocks);


// Name: Dijkstra_wrapper
//
// Description: Wrapper function that executes the Dijkstra kernel
//
//
void Dijkstra_wrapper(int* d_V, int* d_E, short int* d_W, distPred* d_dp,
                      int num_vertices, int num_edges,
                      bool* d_vertex_settled, bool* d_finished, dim3 grid, dim3 blocks);




/* ---- Kernels ---- */

// Name: initialize_GPU_arrays
//
// Description: Initializes the dp and vertex_settled arrays
//
//
__global__ void initialize_GPU_arrays(distPred* d_dp, bool* d_vertex_settled, int infinity,
                                      int source, int num_vertices);

// Name: Dijkstra
//
// Description: Performs the Dijkstra algorithm on the specified vertices
//
//
__global__ void Dijkstra(int* d_V, int* d_E, short int* d_W, distPred* d_dp,
                         int num_vertices, int num_edges,
                         bool* d_vertex_settled, bool* d_finished);


#endif /* KERNELS_H_ */
