

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
// Filename: FW_file_io.h
// Author: Charles W Johnson
// Description: Header file for file I/O for Floyd-Warshall algorithm
//

#ifndef FW_FILE_IO_H_
#define FW_FILE_IO_H_

#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>


using namespace std;


// Name: getFilenameInfo
//
// Description: Determines whether the file is in MTX or CSR format and whether
//              it is a text file or a binary file based on the filename.
//
//
void getFilenameInfo(string graphFile, string& graph_format, string& file_format);


// Name: readInNumVerticesAndEdges
//
// Description: Opens the graph file and reads in enough to discover the number
//              of vertices and edges that the graph contains.
//
//
void readInNumVerticesAndEdges(char* filename, uint32_t& num_vertices, uint32_t& num_edges);


// Function Name: readInGraph
//
// Description: Reads in a standard graph file and store the appropriate values in the
//              dist and (if applicable) next matrices
//
//
void readInGraph(char* filename, uint8_t** graph, int** next);


// Name: saveDistMatrixToDisk
//
// Description: Saves the dist matrix to disk in a binary format
//
//
void saveDistMatrixToDisk(char* input_filename, uint8_t** matrix,
                          uint32_t num_vertices, uint32_t num_edges);


// Name: saveNextMatrixToDisk
//
// Description: Saves the next matrix to disk in a binary format
//
//
void saveNextMatrixToDisk(char* input_filename, int** matrix,
                          uint32_t num_vertices, uint32_t num_edges);


#endif /* FW_FILE_IO_H_ */
