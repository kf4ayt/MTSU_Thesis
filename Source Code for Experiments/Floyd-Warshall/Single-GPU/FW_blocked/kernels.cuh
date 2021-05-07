

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
// Filename: kernels.cuh
// Author: Charles W Johnson
// Description: Header file for wrapper functions and kernels for blocked
//              single GPU-based Floyd-Warshall algorithm
//
// Note: While I edited the kernels and other associated code so as to allow
//       this program to run inside the framework of my thesis programs, credit
//       for the actual implementation goes to Mateusz Bojanowski. I made enough
//       edits to justify adding the MIT license so as to cover my work, but
//       primary credit goes to Mr. Bojanowski. A copy of his implementation
//       can be found at https://github.com/MTB90/cuda-floyd_warshall as of
//       May 6, 2021.
//


#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "constants.h"

using namespace std;
using namespace std::chrono;


/* ---- CPU Wrapper Functions ---- */

void blocked_Dependent_wrapper(uint8_t* d_dist_pc, int* d_next_pc, int num_vertices, int num_rows,
                               int tile_k, int tile_width, int num_tiles_wide, dim3 grid, dim3 blocks);


void blocked_Column_wrapper(uint8_t* d_dist_pc, int* d_next_pc, int num_vertices, int num_rows,
                            int tile_k, int num_tiles_wide, int tile_width, dim3 grid, dim3 blocks);


void blocked_Row_wrapper(uint8_t* d_dist_pc, int* d_next_pc, int num_vertices, int num_rows,
                         int tile_k, int num_tiles_wide, int tile_width, dim3 grid, dim3 blocks);


void blocked_Independent_wrapper(uint8_t* d_dist_pc, int* d_next_pc, int num_vertices, int num_rows,
                                 int tile_k, int num_tiles_wide, int tile_width, dim3 grid, dim3 blocks);


/* ---- GPU Kernels ---- */

__global__ void blocked_Dependent(uint8_t* d_dist_pc, int* d_next_pc, int num_vertices, int num_rows,
                                  int tile_k, int tile_width, int num_tiles_wide);


__global__ void blocked_Column(uint8_t* d_dist_pc, int* d_next_pc, int num_vertices, int num_rows,
                               int tile_k, int num_tiles_wide, int tile_width);


__global__ void blocked_Row(uint8_t* d_dist_pc, int* d_next_pc, int num_vertices, int num_rows,
                            int tile_k, int num_tiles_wide, int tile_width);


__global__ void blocked_Independent(uint8_t* d_dist_pc, int* d_next_pc, int num_vertices, int num_rows,
                                    int tile_k, int num_tiles_wide, int tile_width);


