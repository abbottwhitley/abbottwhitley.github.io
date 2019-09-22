---
layout: post
title: "Implementing a k-Nearest Neighbor classifier"
date: 2019-09-19
category: documentation
comments: true
math: true
author: "J. Abbott"
---

> Self Guided study of the [course notes](http://cs231n.github.io/) for CS231n: Convolutional Neural Networks for Visual Recognition. Course material has been provided as public resource through Stanford University. Various classes and function calls referenced throughout this page were developed by Stanford university as part of the cs231 curriculum. 

In this section, we will implement the k-Nearest Neighbor, (kNN) algorithm for use in a simple image classificaiton system. This system will then be tested against the CIFAR-10 dataset. In addition to this write up, a complete kNN ipython notebook has been recreated, based on the cs231 assignment, in an effort to provide a complete walk through of the various outputs.  



- [Loading the CIFAR-10 dataset](#loading-the-cifar-10-dataset)
- [k-Nearest Neighbor Algorithm](#k-Nearest-Neighbor-Algorithm)
	- [L2 (Euclidean) Distance](#L2-Distance)
    - [Two-Loop Implementation](#twoloop)
    - [One-Loop Implementation](#oneloop)
    - [No-Loop Implementation](#noloop)
- [Cross-validation to find the best _k_](#crossval)


## Loading the CIFAR-10 Dataset

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) is a labeled subset of 60,000 (32x32) color images which were collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinto. The images are categorized within 1 of 10 seperate classifications, with 6,000 images per class. The complete dataset contains 50,000 training images along with 10,000 test images. The test images was created using "exactly 1,000 randomly-selected images from each class". Figure 1 below shows a subsample of 7 images from each class along with the corresponding labels for each class.

![CIFAR Sample](/assets/png/cifar_10_sample.png){:width="560px"}  
__Figure 1:__ _Samples from the CIFAR-10 Dataset_
{: style="text-align: center;"} 

The first step in this process was to simply load the raw CIFAR-10 data into python.  

```python
# Load the raw CIFAR-10 data.
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
# Training images, training labels, test images, test labels
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
```


## <a name="k-Nearest-Neighbor-Algorithm"></a> k-Nearest Neighbor Algorithm

# <a name="twoloop"></a> Two-loop Implementation

# <a name="oneloop"></a> One-loop Implementation

# <a name="noloop"><a/> No-Loop Implementation

## <a name="crossval"><a/> Cross Validation to find the best k-value