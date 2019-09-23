---
layout: post
title: "Implemention of a k-Nearest Neighbor classifier"
date: 2019-09-19
category: documentation
comments: true
math: true
author: "J. Abbott"
---

> Self Guided study of the [course notes](http://cs231n.github.io/) for CS231n: Convolutional Neural Networks for Visual Recognition. Course material has been provided as public resource through Stanford University. Various classes and function calls referenced throughout this page were developed by Stanford university as part of the cs231 curriculum. 

In this section, we will implement the k-Nearest Neighbor algorithm, (k-NN), for use in a simple image classificaiton system which will then be implemented and tested against the CIFAR-10 dataset. In addition to this write up, a Jupyter Notebook of the [k-NN for image classification implementation](/jupyter_notebook/jupyter%20notebooks/2019/09/19/knn_implementation) has been created, compiled and exported as a complete Markdown file in order to provide a complete walk through of the various outputs. Note that the notebook is based on the [cs231 course assignment](http://cs231n.stanford.edu/).

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
Looking at the above output, we can see that the we've loaded all 50,000 (32x32) training images, the 10,000 (32x32) test images, and the corresponding labels for each set. To aid in data processing, a subsample will be created, (python code not shown) so that the process of implementing the k-NN will be more efficient. 

## <a name="k-Nearest-Neighbor-Algorithm"></a> k-Nearest Neighbor Algorithm

The k-nearest neighbors algorithm is a classification method in which the classification of a sample object is determined based on its k-nearest neighbors, where k is a user defined parameter and the classification of the surrounding neighbors is known. It assumes that objects close to each other are similar to each other. The k-NN is defined as a non-parametric method in which the algorithm makes no assumptions about the underlying data sets. It is also an instance-based learning algorithm, (i.e. a lazy algortihm), in which the training data is not used to make any generalizations on the test data set. This means that the "training" step in a k-NN is very short, leaving the majority of the data processing to the classification step. Consequently, this makes the k-NN an inherently poor choice for large data sets. However, the algorithm is simple, easy to implement, and versatile, which makes it a great choice for understanding basic concepts in image classification. 


# <a name="L2-Distance"></a> Calculating L2 (Euclidean) Distance

Knowing that the classification, (i.e. label) of an image can be predicted based on its k-nearest neighbors, a system for comparing images is needed. One method for doing so is to calculate the Euclidean distance, (L2 Distance), between all images within both the test and training data sets. The L2 distance is the "straight-line" distance between any two points and can be calculated as follows. 

$$
d_{2}(I_{1}, I_{2}) = \sqrt{\sum_{p}(I_{1}^{p} - I_{2}^{p})^{2}}
$$

The function above calculates the pixel-wise difference between the two images and then squares the difference. The values of the output matrix is then summed together, from which a final square root is taken. The resulting value provide the L2 distance between image A and image B. Doing this against all images in both the test and training data sets, a final matrix is created which notes the L2 distance between all images. This step for calculating the L2 distance can be thought of as the "training" step for the k-NN.        

# <a name="twoloop"></a> Two-loop Implementation


```python
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        print("Calculating.....")
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                dists[i,j] = (((self.X_train[j,:] - X[i,:])**2).sum())**0.5
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        print("Complete")
        return dists
```
# <a name="oneloop"></a> One-loop Implementation

# <a name="noloop"><a/> No-Loop Implementation

## <a name="crossval"><a/> Cross Validation to find the best k-value