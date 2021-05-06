

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
// Filename: dijkstra-classic-v2.h
// Author: Charles W Johnson
// Description: Header file for C / C++ program for Dijkstra's shortest path
//              algorithm using a CSR representation of the graph
//


#ifndef DIJKSTRA_CLASSIC_V2_H_
#define DIJKSTRA_CLASSIC_V2_H_


#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <stdio.h> 
#include <stdlib.h> 

#include "Dijkstra_custom_data_structures.h"

#define INF 255

using namespace std;


/* ---- Function Declarations ---- */

// Name: dijkstra_classic_cpu
//
// Description: Runs the Dijkstra algorithm on the given graph
//
//
int dijkstra_classic_cpu(int* V, int* E, short int* W, distPred* dp, int num_vertices, int num_edges, int source);


// Name: getMinVertex
//
// Description: Returns the position in Q that contains the vertex with the min distance
//
//
int getMinVertex(vector<int>& Q, distPred* dp);


#endif /* DIJKSTRA_CLASSIC_V2_H_ */
