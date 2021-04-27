

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
// Filename: BF_custom_data_structures.cuh
// Author: Charles W Johnson
// Description: Custom data structures for Bellman-Ford algorithm
//

#ifndef BF_CUSTOM_DATA_STRUCTURES_H_
#define BF_CUSTOM_DATA_STRUCTURES_H_

using namespace std;


// Data structure for an edge
//
struct Edge
{       
    uint32_t u;
    uint32_t v; 
    uint8_t  w;
};


// data structure for dist/pred pair
// 
// Note: To make it easier to handle with MPI, I'm switching to the distPred
//       data structure being the same datatype (int) and to make it possible
//       for the pred value to have a value of -1 (which must be possible for
//       my function to print out the path), I'm going to use signed ints
//       instead of uint32_t.
//
struct distPred
{ 
    int dist;
    int pred;
};


#endif /* BF_CUSTOM_DATA_STRUCTURES_H_ */
