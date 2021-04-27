

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
// Filename: BF_kernels.cuh
// Author: Charles W Johnson
// Description: Header file for kernels for Bellman-Ford algorithm
//

#ifndef BF_KERNELS_H_
#define BF_KERNELS_H_

#include <iostream>

#include "BF_custom_data_structures.cuh"


using namespace std;


// Name: initialize_dp_array
//
// Description: Initializes the d_dp array
//
//
__global__ void initialize_dp_array(distPred* d_dp, int infinity, int source, int num_vertices);


// Name: cudaBellmanFord
//
// Description: Executes the Bellman-Ford algorithm on the specified edges
//
//
__global__ void cudaBellmanFord(uint32_t num_edges, Edge* d_edgeList, distPred* d_dp, int* d_change);
 

#endif /* BF_KERNELS_H_ */
