

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


#include <chrono>
#include <random>

#include <cstdbool>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <time.h>
#include <unistd.h>
#include <vector>


using namespace std;


/* ---- Data structures for graphs ---- */

struct Edge
{
    uint32_t end_vertex;
    uint8_t weight;
};

struct Vertex
{
    uint32_t vertex_number;
    vector<Edge> edgeList;
};


/* ---- Get the random number generator set up ---- */

// obtain a seed from the system clock:
unsigned seed = chrono::system_clock::now().time_since_epoch().count();

mt19937 generator (seed);  // mt19937 is a standard mersenne_twister_engine


/* ---- Function Declarations ---- */

// Returns a random integer between 1 and max_weight (max possible weight is 255)
//
uint8_t genRandomWeight(uint8_t max_weight);


// returns 0 for success, 1 for failure
//
int add_edge(uint32_t source_vertex, uint32_t end_vertex, int degree, uint8_t max_weight, vector<Vertex>& Graph);

// Adds a random edge to a vertex
//
int addRandomEdge(uint32_t source_vertex, uint32_t num_vertices, int degree, uint8_t max_weight, vector<Vertex>& Graph);


// Writes a large graph to a binary file
//
void writeBinaryFile(uint32_t num_vertices, int degree, string file_format, vector<Vertex>& Graph);

// Writes the graph to a text file
//
void writeTextFile(uint32_t num_vertices, int degree, string file_format, vector<Vertex>& Graph);


// Converts a vector graph to CSR arrays
//
void convertGraph(uint32_t num_vertices, uint32_t num_edges, vector<Vertex>& Graph, vector<uint32_t>& V, vector<uint32_t>& E, vector<uint8_t>& W);




/* ---- The main act ---- */

int main(int argc, char* argv[])
{
    // Check for proper number of arguments
    //
    if (argc != 6) {

        cout << "This program takes 5 arguments:" << endl << endl;
        cout << "# of vertices" << endl;
        cout << "vertex degree" << endl;
        cout << "max edge weight (no more than 255)" << endl;
        cout << "MTX or CSR output format (mtx, csr, both)" << endl;
        cout << "Text or binary output file format (txt, bin, both)" << endl << endl;

        return 1;
    }

    /* ---- Transfer CLI arguments to variables ---- */

    uint32_t num_vertices = 0;
    int degree = 0;
    int max_weight_temp = 0;
    uint8_t max_weight = 0;
    string graph_format = "";
    string file_format = "";

    // num_vertices
    //
    num_vertices = atoi(argv[1]);

    if (num_vertices <= 0) {
        cout << "You must have at least 1 vertices" << endl << endl;
        return 1;
    }

    // degree
    //
    degree = atoi(argv[2]);

    if (degree <= 0) {
        cout << "The degree must be greater than 0" << endl << endl;
        return 1;
    }

    // weight
    //
    max_weight_temp = atoi(argv[3]);

    if ((max_weight_temp < 1) || (max_weight_temp > 255)) {
        cout << "The max weight must be between 1 and 255" << endl << endl;
        return 1;
    }

    // if we got this far, assign the max_weight value to max_weight
    //
    max_weight = atoi(argv[3]);

    // graph_format
    //
    graph_format = argv[4];

    if ( !((graph_format == "mtx") || (graph_format == "csr") || (graph_format == "both"))) {
        cout << "Graph format must be 'mtx', 'csr', or 'both'" << endl << endl;
        return 1;
    }

    // graph_format
    //
    file_format = argv[5];

    if ( !((file_format == "txt") || (file_format == "bin") || (file_format == "both"))) {
        cout << "File format must be 'txt', 'bin', or 'both'" << endl << endl;
        return 1;
    }


//
// At this point, we have our graph parameter variables read in and stored.
//

    /* ---- Create the graphs ---- */

    vector<Vertex> Graph;

    Graph.resize(num_vertices);


    cout << "num_vertices is: " << num_vertices << endl;
    cout << "Size of Graph (vertices) is: " << Graph.size() << endl;
    cout << endl;

    cout << "Specified Degree of Graph is: " << degree << endl;
    cout << endl;


    // Set the vertex_numbers for each entry
    //
    for (int i=0; i < num_vertices; i++)
    {
        Graph[i].vertex_number = i;
    }


    // ---- Now, we need to populate the graph ---- //


    // First, we're going to give everyone <degree> shots of getting random vertices
    //
    // (Actually <degree> * 10, as the addRandomEdge function tries 10 times)
    //

    int rc = 0;

    for (int i=0; i < num_vertices; i++)
    {
        /* ---- if the vertex's edgeList is full, move on ---- */

        if (Graph[i].edgeList.size() == degree) {
            continue;
        }


        /* ---- Add random edges ---- */

        int count = 0;

        while (count < degree)
        {
            if (Graph[i].edgeList.size() == degree) {
                break;
            }

            rc = addRandomEdge(i, num_vertices, degree, max_weight, Graph);

            count++;
        }

        count = 0;

        // Status update
        //
        if ((i % 1000) == 0) {
            cout << "Finished with vertex " << i << endl;
        }
    }


    // Before we go any further, as we won't be adding any more edges
    // to the graph, check and see if any vertices have a degree of 0.
    // If so, print a message to STDOUT and exit without writing anything
    // to disk.

    for (int i=0; i < Graph.size(); i++)
    {
        if (Graph[i].edgeList.size() == 0) {
            cout << "There is a vertex of degree 0. Exiting now." << endl << endl;
            return 0;
        }
    }


    // First, though, we create a shortened version of the Graph - just the vertices
    // that aren't full.

    string filename = to_string(num_vertices) + "-vertices_degree-" + to_string(degree) + "_degree_data.txt";
    ofstream degreeFile(filename);

    vector<Vertex> short_Graph;
    vector<uint32_t> short_vertices;
 
    bool failed = false;
    

    degreeFile << endl;
    
    // Now, we're going to print out the graph
    //
    for (int i=0; i < Graph.size(); i++)
    {
        if (Graph[i].edgeList.size() != degree) {

            failed = true;
            short_Graph.push_back(Graph[i]);

            short_vertices.push_back(i);
        }
    }


    if (failed == true) {
        degreeFile << endl;
        cout << endl;

        degreeFile << "not all vertices are of degree " << degree << endl;
        cout << "not all vertices are of degree " << degree << endl;

        degreeFile << endl;
        cout << endl;

        degreeFile << "There are " << short_Graph.size() << " vertices that are not full." << endl;
        cout << "There are " << short_Graph.size() << " vertices that are not full." << endl;

        degreeFile << endl;
        cout << endl;
    }


    // Print out list of non-full vertices
    //
    degreeFile << endl;
    degreeFile << "Non-Full Vertices" << endl;
    degreeFile << "-----------------" << endl;
    degreeFile << endl;

    degreeFile << " " << setw(6) << "Vertex" << "  " << setw(6) << "Degree" << endl;
    degreeFile << " " << setw(6) << "------" << "  " << setw(6) << "------" << endl;
    degreeFile << endl;

    for (int i=0; i<short_Graph.size(); i++)
    {
        degreeFile << " " << setw(6) << short_Graph[i].vertex_number << "  " << setw(6) << short_Graph[i].edgeList.size() << endl;
        degreeFile << endl;
    }

    degreeFile << endl;


    // get average degree

    float average_degree = 0;
    int degree_count = 0;

    for (int i=0; i<Graph.size(); i++)
    {
        degree_count += Graph[i].edgeList.size();
    }

    average_degree = ((float)degree_count / Graph.size());

    degreeFile << "Average degree is: " << average_degree << endl;


    // ---- Write the graph to a file(s) ---- //

    if (graph_format == "mtx")
    {
        if (file_format == "bin")
        {
            cout << "Printing MTX in BIN" << endl;
            writeBinaryFile(num_vertices, degree, graph_format, Graph);
        }

        if (file_format == "txt")
        {
            cout << "Printing MTX in TXT" << endl;
            writeTextFile(num_vertices, degree, graph_format, Graph);
        }

        if (file_format == "both")
        {
            cout << "Printing MTX in BIN and TXT" << endl;
            writeBinaryFile(num_vertices, degree, graph_format, Graph);
            writeTextFile(num_vertices, degree, graph_format, Graph);
        }
    }

    if (graph_format == "csr")
    {
        if (file_format == "bin")
        {
            cout << "Printing CSR in BIN" << endl;
            writeBinaryFile(num_vertices, degree, graph_format, Graph);
        }

        if (file_format == "txt")
        {
            cout << "Printing CSR in TXT" << endl;
            writeTextFile(num_vertices, degree, graph_format, Graph);
        }

        if (file_format == "both")
        {
            cout << "Printing CSR in BIN and TXT" << endl;
            writeBinaryFile(num_vertices, degree, graph_format, Graph);
            writeTextFile(num_vertices, degree, graph_format, Graph);
        }
    }

    if (graph_format == "both")
    {
        if (file_format == "bin")
        {
            cout << "Printing MTX and CSR in BIN" << endl;
            writeBinaryFile(num_vertices, degree, "mtx", Graph);
            writeBinaryFile(num_vertices, degree, "csr", Graph);
        }

        if (file_format == "txt")
        {
            cout << "Printing MTX and CSR in TXT" << endl;
            writeTextFile(num_vertices, degree, "mtx", Graph);
            writeTextFile(num_vertices, degree, "csr", Graph);
        }

        if (file_format == "both")
        {
            cout << "Printing MTX and CSR in BIN and TXT" << endl;
            writeBinaryFile(num_vertices, degree, "mtx", Graph);
            writeBinaryFile(num_vertices, degree, "csr", Graph);
            writeTextFile(num_vertices, degree, "mtx", Graph);
            writeTextFile(num_vertices, degree, "csr", Graph);
        }
    }


    degreeFile.close();

    return 0;
}




/* ---- Function Definitions ---- */


// function to get random weight between 1 and max_weight
//
uint8_t genRandomWeight(uint8_t max_weight)
{
    return (generator() % max_weight + 1);   // a random number between 1 and max_weight
}


// returns 0 for success, 1 for failure
//
int add_edge(uint32_t source_vertex, uint32_t end_vertex, int degree, uint8_t max_weight, vector<Vertex>& Graph)
{
    // First, check and make sure that source_vertex and end_vertex are different
    //
    if (source_vertex == end_vertex)
    {
        return 1;
    }

    // Next, check and see if the source_vertex's edge list is full
    //

    if (Graph[source_vertex].edgeList.size() == degree)
    {
        return 1;
    }

    // Next, check and see if the end_vertex's edge list is full
    //

    if (Graph[end_vertex].edgeList.size() == degree)
    {
        return 1;
    }

    // At this point, we know that we can add an edge between the two vertices
    //

    // Create a temp Edge
    //
    Edge temp;
    temp.end_vertex = end_vertex;
    temp.weight = genRandomWeight(max_weight);

    // Push it onto the edgeList for source_vertex
    //
    Graph[source_vertex].edgeList.push_back(temp);

    // Swap the vertices and repeat
    //
    temp.end_vertex = source_vertex;
    Graph[end_vertex].edgeList.push_back(temp);

    return 0;
}


// Adds a random edge to a vertex
//

// Notes correct?
/*
Note: When getting a random #, if it is equal to the source_vertex OR LESS,
      pitch it straightaway, as all previous vertices will be full.

      Similarly, when hard-coding an edge, when checking to see if the vertex
      is full, there is no point in checking the vertices before the source_vertex,
      as they will ALWAYS be full.
*/

int addRandomEdge(uint32_t source_vertex, uint32_t num_vertices, int degree, uint8_t max_weight, vector<Vertex>& Graph)
{

    // First, a double-check that the edgeList is not full. If so, return 2.
    //
    if (Graph[source_vertex].edgeList.size() == degree)
    {
        return 2;
    }

    // Now, we try to get a usable end_vertex
    //
    int count = 0;
    uint32_t temp_end_vertex = 0;
    int rc = 0;
    bool isPresent = false;
   
    // We're going to give it 10 attempts
    //
    while (count < 10)
    {
        // Check and see if the vertex is full
        //
        if (Graph[source_vertex].edgeList.size() == degree) {
            return 2;
        }

        // Get a random vertex between 0 and num_vertices.
        //
        temp_end_vertex = (generator() % num_vertices);

        // If the generated vertex is the same as the source vertex,
        // increment the counter and loop around.
        //
        if (temp_end_vertex == source_vertex) {
            count++;
            continue;
        }

        // Now, check to see if the temp_end_vertex's edgeList is full.
        //
        if (Graph[temp_end_vertex].edgeList.size() == degree) {
            count++;
            continue;
        }

        // Now, due to the fact that this is a random number, it is possible
        // that it could be for a vertex with which we already share an edge.
        // Thus, we will check and see if it is in our edgeList.
        //
        // Note that we are only checking to see the presence
        // of it AND that the break statement breaks out of the FOR loop,
        // not the overall WHILE loop.
        //
        isPresent = false;

        for (int i=0; i < Graph[source_vertex].edgeList.size(); i++)
        {
            if (Graph[source_vertex].edgeList[i].end_vertex == temp_end_vertex) {
                isPresent = true;
                break;
            }
        }

        // If the temp_end_vertex is already in the source_vertex's edgeList,
        // then increment the counter and use the continue command to skip
        // to the end of the WHILE loop and start again (or not, if count is
        // now 10).
        //
        if (isPresent) {
            count++;
            continue;
        }

        // If we get this far, then we have an end vertex that we can use. Note
        // that we will not test for isPresent being true or false because if
        // it is true, which means that we'd have to loop around, then the test
        // above will have already caught it and looped around.
        //
        // We'll add the edge and then return straight from inside the while loop.
        //
        rc = 0;
        rc = add_edge(source_vertex, temp_end_vertex, degree, max_weight, Graph);

        // If add_edge returns a 0, then all is well, so we return. If it does not,
        // then we are going to return a 1 - an error - and rely on messages printed
        // to STDOUT to indicate what went wrong.
        //
        if (rc == 0) {
            return 0;
        } else {
            return 1;
        }
    }


    // If we get to this point, something is totally screwed up, so print an
    // error message AND return a 1.
    //
    return 1;

    // If the code hits this, we're REALLY in trouble, so print an error message and return a 1.
    //
    cout << "addRandomEdge failed due to size variabe not being one of 2 options" << endl;

    return 1;
}


// Writes the graph to a binary file
//
void writeBinaryFile(uint32_t num_vertices, int degree, string graph_format, vector<Vertex>& Graph)
{
    uint32_t num_edges = 0;

    for (int i=0; i<Graph.size(); i++)
    {
        num_edges += Graph[i].edgeList.size();
    }


    // ---- This is for writing the graph in MTX format ---- //

    if (graph_format == "mtx")
    {
        string filename = to_string(num_vertices) + "-vertices_degree-" + to_string(degree) + "_mtx.bin";
        ofstream fout(filename, ios::binary);

        uint32_t u, v;
        uint8_t  w;

        // First, write the num_vertices and the number of edges, both of which will be ints
        //
        fout.write((char *) &num_vertices, sizeof(num_vertices));  // uint32_t
        fout.write((char *) &num_edges, sizeof(num_edges));        // uint32_t

        // Then, write each edge in MTX format - remember to add 1 to the vertices!
        //
        for (int a=0; a<Graph.size(); a++)
        {
            uint32_t u = Graph[a].vertex_number + 1;
    
            for (int b=0; b<Graph[a].edgeList.size(); b++)
            {
                v = Graph[a].edgeList[b].end_vertex + 1;
                w = Graph[a].edgeList[b].weight;
   
                fout.write((char *) &u, sizeof(uint32_t));  // uint32_t
                fout.write((char *) &v, sizeof(uint32_t));  // uint32_t
                fout.write((char *) &w, sizeof(w));         // uint8_t
            }
        }

        fout.close();
    }


    // ---- This is for writing the graph in CSR format ---- //

    if (graph_format == "csr")
    {
        // ---- Create the arrays ---- //

        vector<uint32_t> V;
        vector<uint32_t> E;
        vector<uint8_t>  W;

        V.resize(num_vertices);
        E.resize(num_edges);
        W.resize(num_edges);

        // ---- Initialize the V, E, and W arrays ---- //

        for (uint32_t a=0; a<num_vertices; a++) V[a] = 0;
        for (uint32_t a=0; a<num_edges; a++) E[a] = 0;
        for (uint32_t a=0; a<num_edges; a++) W[a] = 0;

        // ---- Convert the vector graph to V, E, W arrays ---- //

        cout << "num_vertices: " << num_vertices << endl;
        cout << "num_edges: " << num_edges << endl;
        cout << endl;
        cout << "size of V: " << V.size() << endl;
        cout << "size of E: " << E.size() << endl;
        cout << endl;

        convertGraph(num_vertices, num_edges, Graph, V, E, W);


        string fileName = to_string(num_vertices) + "-vertices_degree-" + to_string(degree) + "_csr.bin";
        ofstream fout(fileName, ios::binary);

        uint32_t v;
        uint32_t e;
        uint8_t  w;

        // First, write the num_vertices and the number of edges, both of which will be ints
        //
        fout.write((char *) &num_vertices, sizeof(num_vertices));  // uint32_t
        fout.write((char *) &num_edges, sizeof(num_edges));        // uint32_t

        // Next, write each vertex's offset number out
        //
        for (int i=0; i < V.size(); i++)
        {
            v = V[i];
            fout.write((char *) &v, sizeof(v));        // uint32_t
        }

        // Then write the E and W arrays, one after the other
        //
        for (uint32_t i=0; i < E.size(); i++)
        {
            e = E[i];

            fout.write((char *) &e, sizeof(e));        // uint32_t
        }

        for (uint32_t i=0; i < W.size(); i++)
        {
            w = W[i];
            fout.write((char *) &w, sizeof(w));        // uint8_t
        }

        fout.close();
    }
}


// Writes the graph to a text file
//
void writeTextFile(uint32_t num_vertices, int degree, string graph_format, vector<Vertex>& Graph)
{
    uint32_t num_edges = 0;

    for (uint32_t i=0; i<Graph.size(); i++)
    {
        num_edges += Graph[i].edgeList.size();
    }


    // ---- This is for writing the graph in MTX format ---- //

    if (graph_format == "mtx")
    {
        string fileName = to_string(num_vertices) + "-vertices_degree-" + to_string(degree) + "_mtx.txt";
        ofstream outFile(fileName);

        outFile << "%%MatrixMarket matrix coordinate integer general" << endl;
        outFile << "%" << endl;
        outFile << to_string(num_vertices) + " " + to_string(num_vertices) + " " + to_string(num_edges) << endl;

        for (int i=0; i<Graph.size(); i++)
        {
            for (int j=0; j<Graph[i].edgeList.size(); j++)
            {
                // Remember to add - Matrix Market files start at 1, not 0
                //
                outFile << to_string(Graph[i].vertex_number + 1) << " " << to_string(Graph[i].edgeList[j].end_vertex + 1) << " " << to_string(Graph[i].edgeList[j].weight) << endl;
            }
        }

        outFile.close();
    }


    // ---- This is for writing the graph in CSR format ---- //

    if (graph_format == "csr")
    {
        // ---- Create the arrays ---- //

        vector<uint32_t> V;
        vector<uint32_t> E;
        vector<uint8_t>  W;

        V.resize(num_vertices);
        E.resize(num_edges);
        W.resize(num_edges);

        // ---- Initialize the V, E, and W arrays ---- //

        for (uint32_t a=0; a<num_vertices; a++) V[a] = 0;
        for (uint32_t a=0; a<num_edges; a++) E[a] = 0;
        for (uint32_t a=0; a<num_edges; a++) W[a] = 0;

        // ---- Convert the vector graph to V, E, W arrays ---- //

        cout << "num_vertices: " << num_vertices << endl;
        cout << "num_edges: " << num_edges << endl;
        cout << endl;
        cout << "size of V: " << V.size() << endl;
        cout << "size of E: " << E.size() << endl;
        cout << endl;

        convertGraph(num_vertices, num_edges, Graph, V, E, W);


        string fileName = to_string(num_vertices) + "-vertices_degree-" + to_string(degree) + "_csr.txt";
        ofstream outFile(fileName);

        // print the num_vertices and num_edges on the same line, separated by a space
        outFile << num_vertices << " " << num_edges << endl;

        // skip a line
        outFile << endl;

        // Print out a list of each vertex, one vertex # per line
        //
        for (int i=0; i < V.size(); i++)
        {
            outFile << V[i] << endl;
        }

        // skip a line
        outFile << endl;

        // Print out the edge #s
        //
        for (uint32_t i=0; i < E.size(); i++)
        {
            outFile << E[i] << endl;
        }

        // skip a line
        outFile << endl;

        // Print out the weights
        //
        for (uint32_t i=0; i < W.size(); i++)
        {
            outFile << (int)W[i] << endl;
        }

        outFile.close();
    }
}


// Converts a vector graph to CSR arrays
//
void convertGraph(uint32_t num_vertices, uint32_t num_edges, vector<Vertex>& Graph,
                  vector<uint32_t>& V, vector<uint32_t>& E, vector<uint8_t>& W)
{
    // ---- Fill the V array ---- //

    // For the V array, which will store offsets for the E/W arrays, V[0] will
    // always be 0, as its edges start at E[0]/W[0]. For V[1] (i=1), the value will be
    // the degree of V[0] or V[i-1]. More broadly, V[i] = V[i-1] + degree for V[i-1].

    uint32_t degree = 0;
    uint32_t E_pos = 0;  // position indicator for where we are in the E/W arrays

    for (int i=0; i < Graph.size(); i++)
    {
        if (i == 0)
        {
            V[i] = 0;
        }
        else
        {
            degree = Graph[i-1].edgeList.size();

            V[i] = V[i-1] + degree;
        }
    }

    // ---- Fill the E and W arrays ---- //

    for (int i=0; i < Graph.size(); i++)
    {
        for (int j=0; j < Graph[i].edgeList.size(); j++)
        {
            E[E_pos] = Graph[i].edgeList[j].end_vertex;
            W[E_pos] = Graph[i].edgeList[j].weight;

            E_pos++;
        }
    }
}


