

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
// Filename: dijkstra-classic-v2.cpp
// Author: Charles W Johnson
// Description: C / C++ program for Dijkstra's shortest path algorithm
//              using a CSR representation of the graph
//

#include <iostream>

#include "dijkstra-classic-v2.h"

using namespace std;


/* ---- Function Definitions ---- */

// Name: dijkstra_classic_cpu
//
// Description: Runs the Dijkstra algorithm on the given graph
//
//
int dijkstra_classic_cpu(int* V, int* E, short int* W, distPred* dp, int num_vertices, int num_edges, int source)
{ 
    // Get memory for the vertex_settled array

    bool* vertex_settled;

    vertex_settled = (bool *) malloc(sizeof(bool) * num_vertices);

    for (int a=0; a<num_vertices; a++) {
        vertex_settled[a] = false;
    }

    // Set distance from the source node - source - to be 0

    dp[source].dist = 0;

    // Now, for the Q or unvisited vertices group, we're going to
    // use a vector, so as to make removing vertices from the group
    // easier.

    vector<int> Q;
    Q.resize(0);

    for (int a=0; a<num_vertices; a++) {
        Q.push_back(a);
    }

    // Now for the guts of the program

    int min_vertex = -1;
    int min_vertex_pos = -1;
    int first_edge;
    int last_edge;
    int temp_v;
    int i;

    while (Q.size() != 0)
    {
        // The first step is to find the vertex in Q with the smallest distance/cost

        // returns the position in Q that contains the min vertex
        min_vertex_pos = getMinVertex(Q, dp);

        // gives us the actual ver
        min_vertex = Q[min_vertex_pos];

        // delete the element in Q that contains the min vertex
        Q.erase(Q.begin()+min_vertex_pos);

        // now, min_vertex is the vertex that was in Q that has the min distance/cost

        first_edge = V[min_vertex];

        if (min_vertex == (num_vertices-1))
        {
            last_edge = (num_edges-1);
        } else {
            last_edge = V[min_vertex+1] - 1;
        }

        i = first_edge;

        while (i <= last_edge)
        {
            temp_v = E[i];      // temp_v contains the vertex at the other end of this edge

            if (vertex_settled[temp_v]) {       // if the vertex has already been settled, increment
                i++;                            // i and move to the next vertex/edge
                continue;
            }

            if ((dp[min_vertex].dist + W[i]) < dp[temp_v].dist)
            {
                dp[temp_v].dist = dp[min_vertex].dist + W[i];
                dp[temp_v].pred = min_vertex;
            }

            i++;
        }

        vertex_settled[min_vertex] = true;
    }


    // Free up memory allocated for arrays
    //
    free(vertex_settled);

    return 0;
}


// Name: getMinVertex
//
// Description: Returns the position in Q that contains the vertex with the min distance
//
//
int getMinVertex(vector<int>& Q, distPred* dp)
{
    // Q simply contains a list of vertices, so if Q[0] = 2, then that simply
    // means that vertex 2 is in Q
    //
    // dist is the weight for each vertex, so for vertex 2 (starting at 0),
    // you would look at dist[2]

    int min_vertex = 0;

    int max_dist = INF;
    int v = 0;
    int v_value = 0;

    for (int a=0; a < Q.size(); a++)
    {
        v = Q[a];  // this should be the vertex in question
        v_value = dp[v].dist;

        if (v_value < max_dist)
        {
            max_dist = v_value;
            min_vertex = a;    // remember, we are returning the -postion- in the Q vector!
        }
    }

    return min_vertex;
}


