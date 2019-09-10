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
img = cv2.imread('tetons.jpg',0)
# Show original image
cv2.imshow("Grand Tetons", img)
# Wait for user to hit key
cv2.waitKey(0)
# Destroy current image window
cv2.destroyAllWindows()

# Radius value for circle
r = 400
# Center x variable of circle
centerX = 450 
# Center Y variable of circle
centerY = 700

# Alpha and Beta values used to adjust contrast on image
alpha = 2
beta = 30
gamma = 0.4

# Loop through all rows and columns in the image
for i in range(np.shape(img)[0]):
    for j in range(np.shape(img)[1]):
        # Calculate the current pixel distance from the circle center point
        d = int(math.sqrt(((i-centerX)**2) + ((j-centerY)**2)))
        # If the current pixel is outside of the circle, adjust the contrast
        if d > r:
            # Adjust contrast of current pixel, (brighten pixel value)
            #img[i, j] = np.clip(alpha * img[i,j] + beta, 0, 255)
            # Gamma correction
            img[i, j] = np.clip(((img[i,j]/255)**gamma) * 255, 0, 255)
        
# Show the updated photo
cv2.imshow("Grand Tetcons", img)  
cv2.waitKey(0)
cv2.destroyAllWindows()        
