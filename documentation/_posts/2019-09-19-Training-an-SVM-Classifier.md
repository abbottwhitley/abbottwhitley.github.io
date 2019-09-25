---
layout: post
title: "Training a Support Vector Machine (SVM)"
date: 2019-09-19
category: documentation
comments: true
math: true
author: "J. Abbott"
---

> Self Guided study of the [course notes](http://cs231n.github.io/) for cs231n: Convolutional Neural Networks for Visual Recognition, provided as a public resource through Stanford University. Included in this page are references to many of the classes and function calls developed by Stanford university as part of the cs231 curriculum. The complete implementation of the [k-NN classifier](/jupyter_notebook/jupyter%20notebooks/2019/09/19/knn_implementation) has been exported as a final Markdown file and can be found in the [Jupyter Notebooks](/jupyter_notebooks/) section of this site.

In this section, we will implement the k-Nearest Neighbor algorithm, (k-NN), for use in a simple image classification system which will then be implemented and tested against the CIFAR-10 dataset. In addition to this write up, a Jupyter Notebook of the 

- [Loading the CIFAR-10 dataset](#loading-the-cifar-10-dataset)
- [k-Nearest Neighbor Algorithm](#k-Nearest-Neighbor-Algorithm)
	- [L2 (Euclidean) Distance](#L2-Distance)
    - [Two-Loop Implementation](#twoloop)
    - [No-Loop Implementation](#noloop)
- [Cross-validation to find the best _k_](#crossval)


## Loading the CIFAR-10 Dataset

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) is a labeled subset of 60,000 (32x32) color images which were collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinto. The images are categorized within 1 of 10 separate classifications, with 6,000 images per class. The complete dataset contains 50,000 training images along with 10,000 test images. The test images was created using "exactly 1,000 randomly-selected images from each class". Figure 1 below shows a subsample of 7 images from each class along with the corresponding labels for each class.

![CIFAR Sample](/assets/png/knn/cifar_10_sample.png){:width="560px"}  
__Figure 1:__ _Samples from the CIFAR-10 Dataset_
{: style="text-align: center;"} 

The first step in this process was to simply load the raw CIFAR-10 data into python. A quick preview of the loaded data is shown in Figure 1 above.   

```python
# Load the raw CIFAR-10 data.
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
# Training images, training labels, test images, test labels
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# View some basic details of the CIFAR-10 data
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
```

```python
Training data shape:  (50000, 32, 32, 3)
Training labels shape:  (50000,)
Test data shape:  (10000, 32, 32, 3)
Test labels shape:  (10000,)
```
Looking at the above output, we can see that we've loaded all 50,000 (32x32) training images, the 10,000 (32x32) test images, and the corresponding labels for each set. To help make the development process more efficient, a subsample of 5,000 training images and 500 test images will be created, (python processing code not shown). 

```python
...
print(X_train.shape, X_test.shape)
```
```python
(5000, 3072) (500, 3072)
```
