#!/usr/bin/env python

##########################################################################
## Imports
##########################################################################

import sys
import argparse
import traceback
import numpy as np

##########################################################################
## Module Constants
##########################################################################

VERSION     = "1.0"
DESCRIPTION = "Generate a random matrix of specified size"
EPILOG      = "This software is for teaching use only."

##########################################################################
## Random Matrix Generator
##########################################################################

def randmatrix(args):
    print (args)
    m = args.m[0]
    n = args.n[0]
    matrix = np.random.choice(10, m*n).reshape((m, n))
    return str(matrix)

##########################################################################
## Main Functionality
##########################################################################

def main(*argv):
    """
    Generates a matrix of given dimensions
    """

    parser     = argparse.ArgumentParser(version=VERSION, description=DESCRIPTION, epilog=EPILOG)
    parser.add_argument('--traceback', action='store_true', default=False, help='On error, show the Python traceback')
    parser.add_argument('m', nargs=1, type=int, help="Number of rows")
    parser.add_argument('n', nargs=1, type=int, help="Number of columns")
    parser.set_defaults(func=randmatrix)

    # Handle input from the command line
    args = parser.parse_args()            # Parse the arguments
    try:
        msg = "%s\n" % args.func(args)    # Call the default function
        parser.exit(0, msg)               # Exit clearnly with message
    except Exception as e:
        if hasattr(args, 'traceback') and args.traceback:
            traceback.print_exc()
        parser.error(str(e))              # Exit with error

if __name__ == "__main__":
    main(sys.argv[1:])