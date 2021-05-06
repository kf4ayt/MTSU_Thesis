

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
// Filename: Dijkstra_print_functions.cpp
// Author: Charles W Johnson
// Description: Print functions for Dijkstra's algorithm
//


#include "Dijkstra_print_functions.h"

using namespace std;


// Name: printResultsPath
//
// Description:  Prints out the cost and the path back to the source
//               vertex for each vertex.
//
//
void printResultsPath(uint32_t num_vertices, uint32_t num_edges, distPred* dp, double duration)
{
    // Open file for recording results
    //
    ofstream resultsFile;
    resultsFile.open("results_path.txt", ios::app);


    // Write # of vertices and edges to it
    //
    resultsFile << endl;
    resultsFile << "# of vertices is: " << num_vertices << endl;
    resultsFile << "# of edges is: " << num_edges << endl;
    resultsFile << endl;


    // Print out the costs and the paths
    //
    int endpoint;

    resultsFile << "Paths (in reverse order):" << endl << endl;

    resultsFile << "Vertex      Cost      Path" << endl;
    resultsFile << "------      ----      ----" << endl << endl;

    for (int a=0; a<num_vertices; a++)
    {
        if (a == 0) {
            resultsFile << setw(6) << a << "    " << setw(6) << dp[a].dist << "      " << a;
            resultsFile << endl;
            continue;
        }

        resultsFile << setw(6) << a << "    " << setw(6) << dp[a].dist << "      " << a;


        // ---- From here on is the path printing part ---- //

        endpoint = dp[a].pred;

        while (endpoint >= 0)
        {
            resultsFile << " -> " << endpoint;

            if (endpoint == 0) {
                endpoint = -1;
            } else {
                endpoint = dp[endpoint].pred;
            }
        }

        resultsFile << endl;
    }

    resultsFile << endl;

    // Print the runtime to the file
    //
    resultsFile << "Time taken by Dijkstra loop: " << (duration / 1000.0) << " milliseconds" << endl;
    resultsFile << "Time taken by Dijkstra loop: " << ((duration / 1000.0) / 1000.0) << " seconds" << endl;
    resultsFile << endl;

    // Close the file
    //
    resultsFile.close();
}


// Name: printResultsCost
//
// Description: Prints out the cost and the predecessor for each vertex.
//
//
void printResultsCost(uint32_t num_vertices, uint32_t num_edges, distPred* dp, double duration)
{
    // Open file for recording results
    //
    ofstream resultsFile;
    resultsFile.open("results_cost.txt", ios::app);


    // Write # of vertices and edges to it
    //
    resultsFile << endl;
    resultsFile << "# of vertices is: " << num_vertices << endl;
    resultsFile << "# of edges is: " << num_edges << endl;
    resultsFile << endl;


    // Print out the costs and the pred
    //
    resultsFile << "Vertex    Cost      Pred" << endl;
    resultsFile << "------    ----      ----" << endl << endl;

    for (int a=0; a<num_vertices; a++)
    {
        resultsFile << setw(6) << a << "    " << setw(4) << dp[a].dist << "    " << setw(6) << dp[a].pred << endl;
    }

    resultsFile << endl;

    // Print the runtime to the file
    //
    resultsFile << "Time taken by Dijkstra loop: " << (duration / 1000.0) << " milliseconds" << endl;
    resultsFile << "Time taken by Dijkstra loop: " << ((duration / 1000.0) / 1000.0) << " seconds" << endl;
    resultsFile << endl;

    // Close the file
    //
    resultsFile.close();
}


// Name: printResultsShort
//
// Description: Prints out the number of vertices and edges and the runtime
//
//
void printResultsShort(uint32_t num_vertices, uint32_t num_edges, double duration)
{
    // Open file for recording results
    //
    ofstream resultsFile;
    resultsFile.open("results_short.txt", ios::app);

    // Write # of vertices and edges to it
    //
    resultsFile << endl;
    resultsFile << "# of vertices is: " << num_vertices << endl;
    resultsFile << "# of edges is: " << num_edges << endl;
    resultsFile << endl;

    // Print the runtime to the file
    //
    resultsFile << "Time taken by Dijkstra loop: " << (duration / 1000.0) << " milliseconds" << endl;
    resultsFile << "Time taken by Dijkstra loop: " << ((duration / 1000.0) / 1000.0) << " seconds" << endl;
    resultsFile << endl;

    // Close the file
    //
    resultsFile.close();
}


