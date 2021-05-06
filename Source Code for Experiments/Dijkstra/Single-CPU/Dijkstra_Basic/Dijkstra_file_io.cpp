

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
// Filename: Dijkstra_file_io.cpp
// Author: Charles W Johnson
// Description: File I/O for Dijkstra's algorithm
//


#include "Dijkstra_file_io.h"


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
void readInNumVerticesAndEdges(char* filename, int& num_vertices, int& num_edges)
{
    num_vertices = 0;   // Number of vertices - initialize to 0
    num_edges = 0;      // Number of vertices - initialize to 0

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


// Name: readInDijstraGraphData
//
// Description: Reads in a standard graph file and returns 3 CSR arrays
//              Note: Will ONLY read in a CSR-formatted graph (this is
//              for simplicity and time)
//
//
int readInDijkstraGraphData(char* filename, int& num_vertices, int& num_edges,
                            int* V, int* E, short int* W)
{

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
            cout << "You must read in from CSR-formatted graph files!" << endl;
            cout << endl;

            fin.close();

            return 1;

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
        

            /* ---- Note that while I am reading the weights into a short int array ---- */
            /* ---- I am reading them in from the file as uint8_ts                  ---- */

            uint32_t v, e;
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
        
            cout << "You must read in from CSR-formatted graph files!" << endl;
            cout << endl;

            inputfile.close();

            return 1;
        
        // end TXT, MTX
        }
        else    // if it's not mtx, then it's csr
        {
            int u, v;
            int  w;

            // In our txt CSR graph files, the first line is 'num_vertices num_edges'
            //
            // The next line is blank, then all the values for V array, a blank line,
            // all the values for the E array, a blank line, and then all the values
            // for the W array.
        
            int linecnt = 0;

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

            inputfile.close();

        } // end TXT, CSR

    } // end TXT

} // end readInDijkstraGraphData


