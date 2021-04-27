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

# The basic idea here is that we take in a graph name, create a batch file that
# runs the MPI-CPU BF program 5 times, catching the output in a file whose name
# is the graph name minus the _csr.bin and plus _X.txt, where X is the number of
# nodes that it ran on. The 5 runtimes will be in
# MICROseconds, one per line.
#
# Another program will then come behind and generate a file that has:
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
# That file will have a filename of X-vertices_degree-Y_neat.txt

#######


#### Function to check and see if anything is in the queue

def checkQueue():

    result = subprocess.run(['squeue'], stdout=subprocess.PIPE).stdout.decode('utf-8')

    result = result[:-1]    # remove the last newline character

    splitResults = result.split('\n')

    return len(splitResults)

#### End function


# A check to make sure that we aren't going to try to read from/write to something that doesn't exist

if (len(sys.argv) != 4):
    print("Correct format is <program name> <input filename> <results dir> <num of nodes>")
    sys.exit()


# create list of graph files
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


    results_dir = sys.argv[2]

    # Now we go into a loop, generating SLURM batch files and running them

    j = 1

    while j <= int(sys.argv[3]):

        batchfile = "#!/bin/bash\n"     # first line
        batchfile += "#\n"              # second line
        batchfile += "#SBATCH --nodes=" + str(j) + "\n"      # setting the # of nodes to use
        batchfile += "#SBATCH --ntasks=" + str(j) + "\n"     # translates to 1 rank/node
        batchfile += "#SBATCH --output=" + results_dir + "/" + graphName + "_" + str(j) + ".txt\n"
        batchfile += "\n"
        batchfile += "mpirun ./MPI-CPU_BF " + graphList[i] + " short no_print no_console no_check\n"
        batchfile += "echo \"\"\n"
        batchfile += "mpirun ./MPI-CPU_BF " + graphList[i] + " short no_print no_console no_check\n"
        batchfile += "echo \"\"\n"
        batchfile += "mpirun ./MPI-CPU_BF " + graphList[i] + " short no_print no_console no_check\n"
        batchfile += "echo \"\"\n"
        batchfile += "mpirun ./MPI-CPU_BF " + graphList[i] + " short no_print no_console no_check\n"
        batchfile += "echo \"\"\n"
        batchfile += "mpirun ./MPI-CPU_BF " + graphList[i] + " short no_print no_console no_check"

        # Now we have a batch file that will run the program on the graph 5 times and store the
        # output in a file named graphName_X.txt where X is the # of nodes and each runtime is
        # stored on its own line.        

        # Now, we need to write it to a temp file
        #
        outfile = open("graph.job", "w")
        outfile.write(batchfile)
        outfile.close()

        # Now, we need to execute it
        #
        result = subprocess.run(['sbatch', 'graph.job'], stdout=subprocess.PIPE).stdout.decode('utf-8')

        # Now we need to spin our wheels and wait for the job to finish. So, every 5 seconds,
        # we're going to check the SLURM queue. If we get back a value of 1, which means that
        # 'squeue' returned only the column headers - 1 line - then we know that the queue is
        # empty

        finished = False

        while (finished == False):

            time.sleep(5)
            num = checkQueue()

            if (num == 1):
                finished = True

        ## end of while


        # Once we're here, we're finished with this many nodes for this graph, so it's time
        # to repeat :-)

        j += 1

    ## End of batch loop

## End for loop


