# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:03:37 2019

@author: abbottjc
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

def my_hist(img, b=256, r=[0, 256]):
    
    bins = np.arange(r[0], r[1], dtype='uint8')
    img_1d = img.ravel()
    hist = np.zeros([b], dtype='uint32')
    for i in range(r[0], r[1]):
        hist[i] = len(np.where(img_1d == i)[0])
    
    return hist, bins
    

def cdf(hist, b=256):
    
    cdf = np.zeros([b], dtype='uint32')
    for i in range(0, b):
        cdf[i] = hist[0:i].sum()
        
    return cdf

# CDF to normalize
# k: total number of pixels
# N: Max value of 256
def cdf_norm(cdf, k, N=256):
    
    sK = np.zeros([k], dtype='uint32')
    for j in range(0, k):
        sK[j] = (cdf[0:j]/N).sum()
        
    return sK


def hist_eq(cdf, size, L=256):
    
    h_eq = np.zeros([L], dtype='uint32')
    for i in range(0, L):
        h_eq[i] = round(((cdf[i]  - min(cdf)) / (size -min(cdf))) * (L-1))
        
    return h_eq

img = cv2.imread('hawkes.jpg',0)
hist, bins = my_hist(img)
cdf = cdf(hist)
cdf_n = cdf_norm(cdf, 256, cdf.sum()/256)
h_eq = hist_eq(cdf, img.size) 



fig = plt.figure(figsize=(15, 4))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)
ax1.bar(bins, hist, color='#007acc', alpha=0.2, linewidth=5)
ax1.set_title('Pixel Density Histogram')
ax2.plot(bins, cdf/1024, color='#007acc', alpha=1, linewidth=0.6)
ax2.set_title('Cumulative Distribution Function Histogram')
ax2.set_xlim([0, 256])
ax3.plot(bins, h_eq, color='#007acc', alpha=1, linewidth=0.6)
ax3.set_title('Equalized Histogram')


cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
cv2.imwrite('hawkes_eq.jpg', img2)



