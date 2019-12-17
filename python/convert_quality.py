#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:59:40 2019

@author: Terence Morley, The University of Manchester, UK
"""

"""
Convert a set of images to jpeg at a set quality.

Usage:
  python3 convert_quality.py 40 1.png 2.jpg
  python3 convert_quality.py 40 *.png

40 is the output jpeg quality.
"""


import sys
import cv2 as cv
import os


if __name__ == '__main__':
    numArgs = len(sys.argv)

    # Check that script executed correctly
    if numArgs < 3:
        print('Error: not enough arguments.')
        print('Usage examples:')
        print('  python3 convert_quality.py 40 1.png 2.jpg')
        print('  python3 convert_quality.py 40 *.png')
        print('Where 40 is the output jpeg quality.')
        sys.exit()

    jpegQuality = int(sys.argv[1])

    # Process each specified file
    for i in range(2, len(sys.argv)):
        pathname = sys.argv[i]
        print('Processing:', pathname)

        # Read a file
        img = cv.imread(pathname)

        # Check if actually read
        if img is None:
            print('Failed to read file:', pathname)
            sys.exit()

        # Create output file name
        (path, filename)  = os.path.split(pathname)
        (name, ext) = os.path.splitext(filename)
        outputFilename = name + '_Q' + str(jpegQuality) + '.jpg'

        # Save file at the specified JPEG quality
        compressionParams = [ cv.IMWRITE_JPEG_QUALITY, jpegQuality]
        cv.imwrite(outputFilename, img, compressionParams)

