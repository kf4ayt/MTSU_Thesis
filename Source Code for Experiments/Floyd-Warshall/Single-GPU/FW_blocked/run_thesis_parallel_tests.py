#!/usr/bin/python3

#######
#
# Runs command line programs and then saves the output to a file
#
#######

import subprocess
import sys


#######


# A check to make sure that we aren't going to try to read from/write to something that doesn't exist

if (len(sys.argv) != 4):
    print("Correct format is <program name> <input filename> <output file name> <num of nodes>")
    sys.exit()


# create list of graph files
#
# opens and closes the INPUT file
# reads in all the lines, removes the newline character, and stores them in the graphList list

with open(sys.argv[1]) as f:
    graphList = f.read().splitlines()


# open the OUTPUT file in APPEND mode
outfile = open(sys.argv[2], "a")

runtimeMicroSec = 0
runtimeMilliSec = 0.000

timeString = ""


a = len(graphList)


# loop over the list
#
for i in list(range(a)):

    # zero out the timeString
    timeString = ""

    # set the graphfilename
    graphfilename = graphList[i]

    # print name of graph file to outfile
    #
    outfile.write(graphfilename + "\n")
    outfile.write("\n")

    j = 1

    while (j <= int(sys.argv[3])):

        timeString = ""         # zero it out

        outfile.write("Number of nodes: " + str(j) + "\n")
        outfile.write("\n")

        for k in range(5):
    
            runtimeMicroSec = 0
            runtimeMilliSec = 0.000
    
            result = subprocess.run(['./blocked_FW', graphfilename, 'next', 'no_save', 'no_console', 'no_check'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    
            runtimeMicroSec = int(result)
            runtimeMilliSec = runtimeMicroSec / 1000
    
            if (k < 4):
                timeString += str(runtimeMilliSec) + ","
            else:
                timeString += (str(runtimeMilliSec) + "  milliseconds\n")

        #### end k for

        outfile.write(timeString + "\n")
        outfile.write("\n")

        j += 1

    #### end j for

    outfile.write("-------\n")
    outfile.write("\n")

#### end i for 


outfile.close()


