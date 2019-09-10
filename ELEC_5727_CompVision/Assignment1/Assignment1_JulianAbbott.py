# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 20:00:24 2019
ELEC 5727 Computer Vision & Image Proc Acceleration
Fall 2019
Assignment 1
@author: abbottjc
"""

import math
import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('tetons_color.jpg')
# Show original image
cv2.imshow("Grand Tetons", img)
# Wait for user to hit key
cv2.waitKey(0)
# Destroy current image window
cv2.destroyAllWindows()

shape = np.shape(img)
mask = np.zeros(shape, dtype = np.uint8)

# Radius value for circle
r = 175
# Center x variable of circle
centerX = 180
# Center Y variable of circle
centerY = 300


# Loop through all rows and columns in the image
for i in range(np.shape(img)[0]):
    for j in range(np.shape(img)[1]):
        # Calculate the current pixel distance from the circle center point
        d = int(math.sqrt(((i-centerX)**2) + ((j-centerY)**2)))
        # If the current pixel is inside of the circle, set pixel value to 1
        if d < r:
            # Create a circle mask
            # Pixels outside of the mask will be given a value of 0
            # Values inside the mask will be given a value of 1
            mask[i, j, :] = 1

# Show / save mask to png
cv2.imshow("Mask", mask * 255)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('mask.png',mask * 255)


        
# Show the clipped photo
cv2.imshow("Grand Tetcons", img * mask)
cv2.waitKey(0)
cv2.destroyAllWindows()        
cv2.imwrite('tetons_clipped.png',img * mask)