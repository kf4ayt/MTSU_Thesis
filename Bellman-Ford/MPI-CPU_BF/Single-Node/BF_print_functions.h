

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
// Filename: BF_file_functions.h
// Author: Charles W Johnson
// Description: Header for print functions for Bellman-Ford algorithm
//

#ifndef BF_PRINT_FUNCTIONS_H_
#define BF_PRINT_FUNCTIONS_H_

#include <iomanip>
#include <iostream>
#include <fstream>

#include "BF_custom_data_structures.h"


using namespace std;


// Name: writeResultFile
//
// Description: Appends to 'results.txt' the # of vertices and edges followed
//              by listing out each vertex, the cost, and then the path. It
//              then prints whether the algorithm was run in short or normal
//              mode, and then the runtime in milliseconds and seconds.
//
//
void writeResultFile(uint32_t num_vertices, uint32_t num_edges, distPred* dp,
                     double duration, bool BF_short);


// Name: writeResultFileShort
//
// Description: Appends to 'results_short.txt' the # of vertices and edges,
//              whether the algorithm was run in short or normal mode, and
//              then the runtime in milliseconds and seconds.
//
//
void writeResultFileShort(uint32_t num_vertices, uint32_t num_edges,
                          double duration, bool BF_short);


#endif /* BF_PRINT_FUNCTIONS_H_ */
