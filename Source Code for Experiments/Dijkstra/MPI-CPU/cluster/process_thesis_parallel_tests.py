#!/usr/bin/python3

#######
#
# Creates a SLURM batch file to run an MPI program, collect the runtimes,
# and store them in a text file.
#
# You feed it in a text file with a list of graph files to evaluate. One
# after the other, it runs them 5 times. For each graph, though, it runs
# it up to and including the number of nodes specified. If the number of
# nodes specified was 4, then it would run it 5 times on 1 node, 5 times
# on 2 nodes, etc.
#
#
#
#######

import subprocess
import sys
import time


#######

#
# Program takes 4 arguments:
#
# 1 - name for file that contains a list of all graphs analyzed
# 2 - directory where the output files are stored
# 3 - # of cluster nodes used
# 4 - # of ranks/node
#

#
# Program outputs 2 types of files:
#
# 1 - Master file that will look like below
#     Filename will be X-vertices_degree-Y_neat.txt
#
# MPI-CPU Bellman-Ford
#
# <graph name>
#
# Nodes  Times
# -----  -----
#
#     1  time1,time2,time3,time4,time5
#
#     2  time1,time2,time3,time4,time5
#
#   ...
#
#   ...
#
# 2 - A file that compresses everything down. Each line will represent X num of nodes
#     and will have the timing for it in 'time1,time2,time3,time4,time5' format so that
#     I can just copy the entire block into Excel.
#     Filename will be X-vertices_degree-Y_times.txt
#

#######


# Now we begin...


# A check to make sure that we aren't going to try to read from/write to something that doesn't exist

if (len(sys.argv) != 5):
    print("Correct format is <program name> <filename of graph names> <result directory> <num of nodes> <num of ranks/node")
    sys.exit()


# create list of graph names
#
# opens and closes the INPUT file
# reads in all the lines, removes the newline character, and stores them in the graphList list

with open(sys.argv[1]) as f:
    graphList = f.read().splitlines()

# We're going to go into one big loop

a = len(graphList)

# For each graph in the list...
#
for i in list(range(a)):

    # First, we remove the '_csr.bin' from a COPY of the filename
    #
    shortName = graphList[i]
    shortName = shortName[:-8]    # remember to count the \n character


    # Next, we want to get just the graph name itself - 'X-vertices_degree-Y'
    #
    slicedFilenameParts = shortName.split('/');
    graphName = slicedFilenameParts[len(slicedFilenameParts)-1]


    # Open two files - the master file and the compressed times file
    #
    result_dir = sys.argv[2]

    neatfile = open((result_dir + "/" + graphName + "_neat.txt"), "w")
    compfile = open((result_dir + "/" + graphName + "_times.txt"), "w")


    # Veer off and print the opening for the neat file
    #
    ranksPerNode = sys.argv[4]

    neatfile.write("\n")
    neatfile.write("MPI-CPU Bellman-Ford\n")
    neatfile.write("\n")
    neatfile.write(graphName + "\n")
    neatfile.write("\n")
    neatfile.write("Ranks/Node: " + ranksPerNode + "\n")
    neatfile.write("\n")
    neatfile.write("Nodes  Times\n")
    neatfile.write("-----  -----\n")
    neatfile.write("\n")

    # At this point, we're ready to begin reading and writing the times

    node_count = int(sys.argv[3])

    # Now we need to go through each node count file

    j = 1

    while (j <= node_count):

        # First, we form the filename that we're looking for
        #
        timeFilename = result_dir + "/" + graphName + "_" + str(j) + ".txt"

        # Now, we open/close it and read in the times
        #
        with open(timeFilename) as f:
            microTimeList = f.read().splitlines()

        # Now, we want to convert the times in the microseconds list to milliseconds (to 3 deciment points)
        # and create a string with the times in them.
        #
        # Form the time string
        #
        timeString = ""

        # We know that there will only be 5 time entries, so...
        #
        k = 0

        while (k < 5):

            microTime = microTimeList[k]
            timeLen = len(microTimeList[k])

            # Slices the string such that I can simply put a '.' to the left of the last 3 digits
            #
            tempTime = microTime[:(timeLen-3)] + "." + microTime[(timeLen-3):] 

            if (k < 4):
                timeString += str(tempTime) + ","
            else:
                timeString += str(tempTime)

            k += 1

        ## End of micro to milli while loop


        # We now have our timeString, so we need to write to the output files
        #
        neatfile.write(str(j).rjust(5) + "  " + timeString + "\n\n")
        compfile.write(timeString + "\n")

        # And that is that

        j += 1

    ## End of node_count loop


    # Close out the output files
    #
    neatfile.close()
    compfile.close()

## End for loop


