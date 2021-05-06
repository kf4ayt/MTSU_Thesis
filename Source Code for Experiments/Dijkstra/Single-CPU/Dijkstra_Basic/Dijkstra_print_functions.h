

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
// Filename: Dijkstra_print_functions.h
// Author: Charles W Johnson
// Description: Header file for print functions for Dijkstra's algorithm
//

#ifndef DIJKSTRA_PRINT_FUNCTIONS_H_
#define DIJKSTRA_PRINT_FUNCTIONS_H_

#include <iomanip>
#include <iostream>
#include <fstream>

#include "Dijkstra_custom_data_structures.h"

using namespace std;


// Name: printResultsPath
//
// Description:  Prints out the cost and the path back to the source
//               vertex for each vertex.
//
//
void printResultsPath(uint32_t num_vertices, uint32_t num_edges, distPred* dp, double duration);


// Name: printResultsCost
//
// Description: Prints out the cost and the predecessor for each vertex.
//
//
void printResultsCost(uint32_t num_vertices, uint32_t num_edges, distPred* dp, double duration);


#endif /* DIJKSTRA_PRINT_FUNCTIONS_H_ */
