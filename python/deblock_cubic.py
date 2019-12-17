#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:53:58 2019

@author: Terence Morley, The University of Manchester, UK
"""

import numpy as np
import cv2 as cv
import math
import csv
import time
import sys

# Globals
SCALE = 1
INTERPOLATION = cv.INTER_CUBIC
ROI_MAX_DIST = 300




#<<<<<<<<<<<<<<<< CLASS: SuperRes <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class SuperRes(object):
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    def __init__(self):
        # List of images - last one is the reference image
        self.image_name_list = []
        self.Hinv_list = []
        self.imAcc = None # will be image for accumulating other images
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    def registerImages(self, img1, img2, poi=None):
        # poi = point of interest (row, col)
               
        # Initiate feature detector
        fd = cv.ORB_create(nfeatures=5000)
        #fd = cv.BRISK_create()
        
        # find the keypoints and descriptors with ORB
        kp1, des1 = fd.detectAndCompute(img1, None)
        kp2, des2 = fd.detectAndCompute(img2, None)
        
        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        
        # Match descriptors.
        matches = bf.match(des1, des2)
        
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        
        # Sort them in the order of the kepoint distance to the poi.
        close_by = []
        if poi is not None:
            for m in matches:
                d = math.sqrt( (poi[0] - kp2[m.trainIdx].pt[0]) ** 2.0 + (poi[1] - kp2[m.trainIdx].pt[1]) ** 2.0 )
                if d < ROI_MAX_DIST:
                    close_by += [ m ]
        
        # Draw first 100 matches.
        img_matches=img1.copy()
        img_matches = cv.drawMatches(img1, kp1, img2, kp2, matches, img_matches, flags=2)

        # Not doing a ratio test here - the docs say crossCheck in BFMatcher does a similar job
        #good = matches#[:50]
        if poi is None:
            good = matches
        else:
            good = close_by
        
        # Get the corresponding points for the good matches
        src_points = [kp2[good[i].trainIdx].pt for i in range(len(good))]
        dst_points = [kp1[good[i].queryIdx].pt for i in range(len(good))]
        
        img_kp = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
        for pt in src_points:
            cv.circle(img_kp, (int(pt[0]),int(pt[1])), 2, (255,0,0), 4)
        
        # Find Homography
        Hinv, _ = cv.findHomography(np.array(src_points), np.array(dst_points), cv.RANSAC)
        
        return Hinv
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    def warpImages(self, scale=1):
        # Output image is of specified size but the image is zoomed with scale
        last_index = len(self.image_name_list) - 1

        im_ref = cv.imread(self.image_name_list[last_index])

        # Check if actually read
        if im_ref is None:
            print('Failed to read file:', self.image_name_list[last_index])
            sys.exit()

        rows = im_ref.shape[0] * scale
        cols = im_ref.shape[1] * scale

        # Scaling matrix
        M = np.array([[scale,0,0],[0,scale,0],[0,0,1]])

	# Create accumulation image
        if self.imAcc is None:
            self.imAcc = np.zeros((rows, cols, 3), np.float)

        img_out = np.zeros((rows, cols), np.float)
        row_limit = rows - 1
        col_limit = cols - 1

        # Loop through all images warping and add proportions to destination pixels
        for i in range( len(self.image_name_list) ):
            im = cv.imread(self.image_name_list[i])

            # Check if actually read
            if im is None:
                print('Failed to read file:', self.image_name_list[i])
                sys.exit()

            print('Warping image ', i)
            H = M @ self.Hinv_list[i]

            im = cv.warpPerspective(im, H, (cols, rows), flags=INTERPOLATION)
            self.imAcc += im

        return
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    def combineImages(self):
        self.imAcc = self.imAcc / len(self.image_name_list)
        im = self.imAcc.astype(np.uint8)

        return im
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    def registerImagesToRef(self, poi=None):
        last_index = len(self.image_name_list) - 1
        
        self.Hinv_list = []

        for i in range(last_index):
            print('Registering image', i)
            im_ref = cv.imread(self.image_name_list[last_index], cv.IMREAD_GRAYSCALE)

            # Check if actually read
            if im_ref is None:
                print('Failed to read file:', self.image_name_list[last_index])
                sys.exit()

            im = cv.imread(self.image_name_list[i], cv.IMREAD_GRAYSCALE)

            # Check if actually read
            if im is None:
                print('Failed to read file:', self.image_name_list[i])
                sys.exit()

            Hinv = self.registerImages(im_ref, im,  poi )
            self.Hinv_list += [ Hinv ]

        # NOTE just use the identity matrix for the ref img
        self.Hinv_list += [ np.eye(3) ]

        print("Done registering")
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#>>>>>>>>>>>>>>>> CLASS: SuperRes >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>




#== Main =====================================================
if __name__ == '__main__':

    # Create Super-resolution object
    res = SuperRes()

    # Read images specified in config.txt. The last one is the reference image.
    with open('config.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) < 2:
                continue
            if row[0] == 'img':
                print(row[1])
            res.image_name_list += [row[1]]

    # Process images:
    print('Registering images.')
    start = time.time()
    res.registerImagesToRef()
    end = time.time()
    print("Finished registering images: %s" % (end - start))

    print('Warping images.')
    start = time.time()
    res.warpImages(SCALE)
    end = time.time()
    print("Finished warping images: %s" % (end - start))

    print('Combining images.')
    start = time.time()
    im = res.combineImages()
    end = time.time()
    print('Finished combining images: %s'% (end - start))

    # Save output file
    cv.imwrite( ("IM%d_%dx.jpg" % (len(res.image_name_list),  SCALE)), im )

