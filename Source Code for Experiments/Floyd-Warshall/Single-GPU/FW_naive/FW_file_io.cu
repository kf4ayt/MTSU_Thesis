

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
// Filename: FW_file_io.cu
// Author: Charles W Johnson
// Description: File I/O for Floyd-Warshall algorithm
//


#include "FW_file_io.cuh"


using namespace std;


/* ---- CPU File I/O Functions ---- */

// Name: getFilenameInfo
//
// Description: Determines whether the file is in MTX or CSR format and whether
//              it is a text file or a binary file based on the filename.
//
//
void getFilenameInfo(string graphFile, string& graph_format, string& file_format)
{
    /* ---- Find out from the file name whether or not it's bin/txt and mtx/csr ---- */

    size_t result;

    // ---- type - mtx or csr ---- //

    // see if it's mtx    
    result = graphFile.find("mtx", 0);

    if (result != string::npos) {
        graph_format = "mtx";
    }

    // see if it's csr
    result = graphFile.find("csr", 0);

    if (result != string::npos) {
        graph_format = "csr";
    }

    // ---- file format - bin or txt ---- //

    // see if it's bin    
    result = graphFile.find("bin", 0);

    if (result != string::npos) {
        file_format = "bin";
    }

    // see if it's txt
    result = graphFile.find("txt", 0);

    if (result != string::npos) {
        file_format = "txt";
    }
}


// Name: readInNumVerticesAndEdges
//
// Description: Opens the graph file and reads in enough to discover the number
//              of vertices and edges that the graph contains.
//
//
void readInNumVerticesAndEdges(char* filename, uint32_t& num_vertices, uint32_t& num_edges)
{
    num_vertices = 0;   // Number of vertices - initialize to 0
    num_edges = 0;      // Number of edges - initialize to 0

    /* ---- Find out from the file name whether or not it's short/int, bin/txt, and mtx/csr ---- */

    string graphFile = filename;
    string graph_format = "";
    string file_format = "";

    getFilenameInfo(graphFile, graph_format, file_format);


    /* ---- Choose between file formats - txt or bin ---- */

    // If the file is a binary file, we'll process it a certain way
    //
    if (file_format == "bin")
    {

        /* ---- Choose between graph formats - mtx or csr ---- */

        // If the format is mtx...
        //
        // Note: For the binary MTX graph files generated for this program,
        //       it simply begins with:
        //
        //       num_vertices   // uint32_t
        //       num_ediges     // uint32_t
        //       u v w          // whether they are 16- or 32-bit depends upon the file name
        //
        if (graph_format == "mtx")
        {
            // Open the binary file and then read it in as if was an MTX text file

            // now open the contents of the file
            ifstream fin(filename, ios::binary);

            // get num_vertices
            fin.read(reinterpret_cast<char *>(&num_vertices), sizeof(num_vertices));  // uint32_t
    
            // get num_edges
            fin.read(reinterpret_cast<char *>(&num_edges), sizeof(num_edges));  // uint32_t
    
            fin.close();
        }
        else    // if it's not mtx, then it's csr
        {
            // Open the binary file and then read it in as if was a CSR text file
            //
            // Note: For our binary CSR graphs, the first line is the # of vertices as a uint32_t

            // now open the contents of the file
            ifstream fin(filename, ios::binary);

            // get num_vertices
            fin.read(reinterpret_cast<char *>(&num_vertices), sizeof(num_vertices));  // uint32_t

            // get num_edges
            fin.read(reinterpret_cast<char *>(&num_edges), sizeof(num_edges));  // uint32_t

            fin.close();
        }
    }
    else        // since we know that it can only be 1 of 2 options, if it's not 'bin', then it must be 'txt'
    {

        /* ---- Choose between graph formats - mtx or csr ---- */

        // If the format is mtx...
        //
        // Note: For these txt MTX files, they begin with the standard Market Matrix header
        //
        if (graph_format == "mtx")
        {
            // Open the .mtx input file
            //
            ifstream inputfile;
        
            inputfile.open(filename);
        
            // We are going to read in the first 2 lines and ignore
            // them. For the third line, we are going to get the
            // first number on the left, which is the # of vertices,
            // and then set V equal to that number.
        
            int linecnt = 1;
            string line;
            string word;
        
            while ((getline(inputfile, line)) && (linecnt < 4))
            {
                if (linecnt == 1) {
                    linecnt++;
                    continue;
                }
        
                if (linecnt == 2) {
                    linecnt++;
                    continue;
                }
        
                if (linecnt == 3) {
                    linecnt++;
        
                    istringstream ss(line);

                    ss >> word;
                    num_vertices = stoi(word);

                    ss >> word; // do nothing with the middle value

                    ss >> word;
                    num_edges = stoi(word);
                }
            }
        
            inputfile.close();
        }
        else    // if it's not mtx, then it's csr
        {
            // Open the graph file and read it in
            //
            // Note: For these txt CSR files, the first line is 'num_vertices num_edges'

            // Open the .csr input file
            //
            ifstream inputfile;
        
            inputfile.open(filename);

            string line = "";
            string word = "";

            // read in the first line
            getline(inputfile, line);
            
            istringstream ss(line);

            ss >> word;
            num_vertices = stoi(word);

            ss >> word;
            num_edges = stoi(word);

            inputfile.close();
        }
    }
}


// Function Name: readInGraph
//
// Description: Reads in a standard graph file and store the appropriate values in the
//              dist and (if applicable) next matrices
//
void readInGraph(char* filename, uint8_t** graph, int** next, bool use_next)
{
    uint32_t num_vertices = 0;   // Number of vertices - initialize to 0
    uint32_t num_edges = 0;      // Number of edges - initialize to 0

    /* ---- Find out from the file name whether or not it's bin/txt and mtx/csr ---- */

    string graphFile = filename;
    string graph_format = "";
    string file_format = "";

    getFilenameInfo(graphFile, graph_format, file_format);


    /* ---- Choose between file formats - txt or bin ---- */

    // If the file is a binary file, we'll process it a certain way
    //
    if (file_format == "bin")
    {
        // ---- First, read in the num_vertices and num_edges - for binary files, it is the same, mtx or csr ---- //

        // open file for reading
        //
        ifstream fin(filename, ios::binary);

        // get num_vertices
        //
        fin.read(reinterpret_cast<char *>(&num_vertices), sizeof(num_vertices));  // uint32_t

        // get num_edges
        //
        fin.read(reinterpret_cast<char *>(&num_edges), sizeof(num_edges));        // uint32_t


        /* ---- Choose between graph formats - mtx or csr ---- */

        // If the format is mtx...
        //
        if (graph_format == "mtx")
        {
            // Open the binary file and then read it in as if was an MTX text file
            //
            // Note: For the binary MTX graph files generated for this program,
            //       it simply begins with:
            //
            //       num_vertices   // uint32_t
            //       num_edges      // uint32_t
            //       u v w          // uint32_t uint32_t uint8_t
            //

            uint32_t u, v;
            uint8_t  w;

            uint32_t count = 0;

            count = 0;
    
            // Since this is MTX, remember to subtract 1 for the vertices
            //
            while (count < num_edges)
            {
                fin.read(reinterpret_cast<char *>(&u), sizeof(u));
                u = (u-1);

                fin.read(reinterpret_cast<char *>(&v), sizeof(v));
                v = (v-1);

                fin.read(reinterpret_cast<char *>(&w), sizeof(w));

                graph[u][v] = w; 

                if (use_next) {
                    next[u][v] = v; 
                }

                count++;
            }

            fin.close();

        } // end BIN, MTX
        else    // if it's not mtx, then it's csr
        {
            // Open the binary file and then read it in as if was a CSR text file

            // open file for reading
            ifstream fin(filename, ios::binary);

            // get num_vertices
            fin.read(reinterpret_cast<char *>(&num_vertices), sizeof(num_vertices));  // uint32_t
        
            // get num_edges
            fin.read(reinterpret_cast<char *>(&num_edges), sizeof(num_edges));        // uint32_t
        

            // ---- Create receiving arrays ---- //
    
            vector<uint32_t> V;
            vector<uint32_t> E;
            vector<uint8_t> W;
    
            // resize the arrays
            //
            V.resize(num_vertices);
            E.resize(num_edges);
            W.resize(num_edges);
     

            uint32_t u, v;
            uint32_t e;
            uint8_t  w;
    
            uint32_t count = 0;

            while (count < num_vertices)
            {
                fin.read(reinterpret_cast<char *>(&v), sizeof(v));      // uint32_t
                V[count] = v;
                count++;
            }

            count = 0;

            while (count < num_edges)
            {
                fin.read(reinterpret_cast<char *>(&e), sizeof(e));      // uint32_t
                E[count] = e;
                count++;
            }

            count = 0;

            while (count < num_edges)
            {
                fin.read(reinterpret_cast<char *>(&w), sizeof(w));      // uint8_t
                W[count] = w;
                count++;
            }


            // store things in the graph

            uint32_t offset;
            uint32_t degree;

            for (int i=0; i < num_vertices; i++)
            {
                offset = V[i];

                if (i == num_vertices - 1)
                {
                    degree = num_edges - V[i];
                }
                else
                {
                    degree = V[i+1] - V[i];
                }

                for (int j=0; j<degree; j++)
                {
                    u = i;
                    v = E[offset + j];
                    w = W[offset + j];

                    graph[u][v] = w;

                    if (use_next) {
                        next[u][v] = v;
                    }
                }
            }

            fin.close();

        } // end BIN, CSR

    } // end BIN
    else        // since we know that it can only be 1 of 2 options, if it's not 'bin', then it must be 'txt'
    {

        /* ---- Choose between graph formats - mtx or csr ---- */

        // ---- First, open up the file ---- //

        // Open the .mtx input file
        //
        ifstream inputfile;
        
        inputfile.open(filename);


        // If the format is mtx...
        //
        if (graph_format == "mtx")
        {
        
            // We are going to read in the first 2 lines and ignore
            // them. For the third line, we are going to get the
            // first number on the left, which is the # of vertices,
            // and then set V equal to that number.
        
            int linecnt = 1;
            string line;
            string word;

            while ((getline(inputfile, line)) && (linecnt < 4))
            {
                if (linecnt == 1) {
                    linecnt++;
                    continue;
                }
        
                if (linecnt == 2) {
                    linecnt++;
                    continue;
                }
        
                if (linecnt == 3) {
                    linecnt++;
        
                    istringstream ss(line);
                    ss >> word;
        
                    num_vertices = stoi(word);
                }
            }

            /* ---- Read in the edges and map the weights to the graph ---- */
        
            // temp variables
        
            uint32_t u, v;
            uint8_t weight;
        
            /* ---- Important Note! ---- */
        
            /* Remember that the .mtx file vertices start at -1-, not 0!!! */
            /* Accordingly, we have to subtract one for all of the vertices */
        
            // Now, currently, the first edge of the graph (the 4th line of the graph file)
            // is in the variable 'line', so first process it.
        
            istringstream single_ss(line);
            string temp;
        
            for (int a=0; a<3; a++) {
                single_ss >> temp;
        
                if (a == 0) u = (stoi(temp) - 1);
                if (a == 1) v = (stoi(temp) - 1);
                if (a == 2) weight = stoi(temp);
        
            }
        
            // Put the data in the graph
            //
            graph[u][v] = weight;

            if (use_next) {
                next[u][v] = v;
            }


            // Now process the rest of the file
        
            while (getline(inputfile, line))
            {
                istringstream ss(line);
        
                for (int a=0; a<3; a++) {
                    ss >> temp;
        
                    if (a == 0) u = (stoi(temp) - 1);
                    if (a == 1) v = (stoi(temp) - 1);
                    if (a == 2) weight = stoi(temp);
                }
        
                graph[u][v] = weight;

                if (use_next) {
                    next[u][v] = v;
                }
            }
        
        // end TXT, MTX
        }
        else    // if it's not mtx, then it's csr
        {

            // ---- Create receiving arrays ---- //
    
            vector<uint32_t> V;
            vector<uint32_t> E;
            vector<uint8_t> W;


            uint32_t u, v;
            uint8_t  w;

            // In our txt CSR graph files, the first line is 'num_vertices num_edges'
            //
            // The next line is blank, then all the values for V array, a blank line,
            // all the values for the E array, a blank line, and then all the values
            // for the W array.
        
            uint32_t linecnt = 0;

            string line;
            string word;

            // Get num_vertices and num_edges
            //
            getline(inputfile, line);

            istringstream ss(line);

            ss >> word;
            num_vertices = stoi(word);

            ss >> word;
            num_edges = stoi(word);
       

            V.resize(num_vertices);    
            E.resize(num_edges);    
            W.resize(num_edges);    
  
            // gets the blank line 
            getline(inputfile, line);

            linecnt = 0;
 
            while ((getline(inputfile, line)) && (linecnt < num_vertices))
            {
                istringstream ss(line);
                ss >> word;
        
                V[linecnt] = stoi(word);

                linecnt++;
            }
        
            /* ---- Read in the edges and map the weights to the graph ---- */

            linecnt = 0;

            while ((getline(inputfile, line)) && (linecnt < num_edges))
            {
                istringstream ss(line);
        
                ss >> word;
                E[linecnt] = stoi(word);

                linecnt++;
            }
      
 
            linecnt = 0;

            while ((getline(inputfile, line)) && (linecnt < num_edges))
            {
                istringstream ss(line);
        
                ss >> word;
                W[linecnt] = stoi(word);

                linecnt++;
            }
       
 
            /* ---- Fill the graph with the data in the vectors ---- */
        
            // store things in the graph

            uint32_t offset;
            uint32_t degree;

            for (int i=0; i < num_vertices; i++)
            {
                offset = V[i];

                if (i == num_vertices - 1)
                {
                    degree = num_edges - V[i];
                }
                else
                {
                    degree = V[i+1] - V[i];
                }

                for (int j=0; j<degree; j++)
                {
                    u = i;
                    v = E[offset + j];
                    w = W[offset + j];

                    graph[u][v] = w;

                    if (use_next) {
                        next[u][v] = v;
                    }
                }
            }

        } // end TXT, CSR

        // close the input file
        inputfile.close();

    } // end TXT

} // end readInGraph


// Name: saveDistMatrixToDisk
//
// Description: Saves the dist matrix to disk in a binary format
//
void saveDistMatrixToDisk(char* input_filename, uint8_t** matrix,
                          uint32_t num_vertices, uint32_t num_edges)
{
    string in_filename = input_filename;
    string output_filename = in_filename + "_dist.bin";
    ofstream fout(output_filename, ios::binary);

    fout.write((char *) &num_vertices, sizeof(num_vertices));   // uint32_t
    fout.write((char *) &num_edges, sizeof(num_edges));         // uint32_t

    uint8_t n = 0;

    for (int i=0; i < num_vertices; i++)
    {
        for (int j=0; j < num_vertices; j++)
        {
            n = matrix[i][j];
            fout.write((char *) &n, sizeof(n));         // uint8_t
        }
    }

    fout.close();
}


// Name: saveNextMatrixToDisk
//
// Description: Saves the next matrix to disk in a binary format
//
void saveNextMatrixToDisk(char* input_filename, int** matrix,
                          uint32_t num_vertices, uint32_t num_edges)
{
    string in_filename = input_filename;
    string output_filename = in_filename + "_next.bin";
    ofstream fout(output_filename, ios::binary);

    fout.write((char *) &num_vertices, sizeof(num_vertices));   // uint32_t
    fout.write((char *) &num_edges, sizeof(num_edges));         // uint32_t

    int n = 0;

    for (int i=0; i < num_vertices; i++)
    {
        for (int j=0; j < num_vertices; j++)
        {
            n = matrix[i][j];
            fout.write((char *) &n, sizeof(n));         // int
        }
    }

    fout.close();
}


