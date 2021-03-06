#!/usr/bin/env python

import cv2
import numpy
import sys
import os

from PointCloudApp import *

# Get command line arguments or print usage and exit
if len(sys.argv) > 2:
    proj_file = sys.argv[1]
    cam_file = sys.argv[2]
else:
    progname = os.path.basename(sys.argv[0])
    print >> sys.stderr, 'usage: '+progname+' PROJIMAGE CAMIMAGE'
    sys.exit(1)

# Load in our images as grayscale (1 channel) images
proj_image = cv2.imread(proj_file, cv2.IMREAD_GRAYSCALE)
cam_image = cv2.imread(cam_file, cv2.IMREAD_GRAYSCALE)

# Make sure they are the same size.
assert(proj_image.shape == cam_image.shape)

# Set up parameters for stereo matching (see OpenCV docs at
# http://goo.gl/U5iW51 for details).
min_disparity = 0
max_disparity = 16
# window_size = 11
window_size = 15

param_P1 = 0
param_P2 = 20000

# Create a stereo matcher object
matcher = cv2.StereoSGBM_create(min_disparity,
                                max_disparity,
                                window_size,
                                param_P1,
                                param_P2)

# Compute a disparity image. The actual disparity image is in
# fixed-point format and needs to be divided by 16 to convert to
# actual disparities.
disparity = matcher.compute(cam_image, proj_image) / 16.0

# Pop up the disparity image.
cv2.imshow('Disparity', disparity/disparity.max())
while cv2.waitKey(1) < 0: pass

# Create the calibration matrix K
# K = numpy.array([ [[600,0,320]], [[0,600,240]], [[0,0,1]] ])
# print "This is K"
# print K
# print "\n"

# Put into Matrix
# K = numpy.matrix(K)
# print "Matrix Form K"
# print K
# print "\n"

# Get K Inverse
# K_Inverse = K.I
# print "K Inverse"
# print K_Inverse
# print "\n"

# Ignore K Stuff. Don't need.

### Linear Equation Form ###

# Height and Width from Disparity
height, width = disparity.shape

# Domain and Range
Domain = numpy.linspace(0, width-1, width).astype('float32')
Range = numpy.linspace(0, height-1, height).astype('float32')
# print Domain, Range

# Meshgrid
X, Y = numpy.meshgrid(Domain, Range)
X = (X-320)/600
Y = (Y-240)/600

# Same Attributes as X
Z = numpy.copy(X)

# Stereo Baseline
b = 0.05

# Set Zmax
Zmax = 8

# Create Mask
# mask = numpy.logical_and(disparity < (b*600)/Zmax) # Not this one
mask = numpy.logical_not(disparity < (b*600)/Zmax)

# Compute Z values for Torus
Z[mask] = (.05*600)/disparity[mask]

# Scale to Correct Size (Shorter)
X = Z*X
Y = Z*Y

# Fill in the array for only valid XY locations
array = numpy.hstack((X[mask].reshape((-1,1)),
                      Y[mask].reshape((-1,1)),
                      Z[mask].reshape((-1,1))))

# Create Cloud
# cloud = PointCloudApp(array, allow_opengl=allow_opengl)
cloud = PointCloudApp(array)
cloud.run()
