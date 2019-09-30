---
layout: post
title: "Multiclass Support Vector Machine (SVM)"
date: 2019-09-26
category: documentation
comments: true
math: true
author: "J. Abbott"
---

> Self Guided study of the [course notes](http://cs231n.github.io/) for cs231n: Convolutional Neural Networks for Visual Recognition, provided as a public resource through Stanford University. Included in this page are references to many of the classes and function calls developed by Stanford university as part of the cs231 curriculum. The complete implementation of the [SVM classifier](/jupyter_notebook/jupyter%20notebooks/2019/09/19/knn_implementation) has been exported as a final Markdown file and can be found in the [Jupyter Notebooks](/jupyter_notebooks/) section of this site.

In this section, we continue the task of image classification on the CIFAR-10 labeled dataset. In this approach we'll take a linear classification approach using the Support Vector Machine, (SVM), a supervised learning algorithm. The model outputs a class identity by use of a linear function which assumes a boundary exists that separates one class boundary from another. The primary goal of the SVM is to efficiently find the boundary which separates one class from another. In this implementation, we will utilize a linear score function to compute a class score for the input data set. The output score can then be used within a loss function to better determine the success of the linear score function. Stochastic Gradient Descent will then be utilized as the optimization algorithm to minimize the loss determined by the loss function. 

- [Loading the CIFAR-10 dataset and Pre-Processing](#loading-the-cifar-10-dataset)
- [Data Pre-Processing](#pre-processing)
- [Linear Classification](#Linear-Classification)
	- [Linear Score Function](#Linear-Score-Function)
	- [Loss Function](#Loss-Function)
    - [Multiclass Support Vector Machine (SVM)](#SVM)
    - [Stochastic Gradient Descent](#SGD)
- [Hyperparameter Tuning and Cross Validation](Hyperparameter-tuning)

## Loading the CIFAR-10 Dataset

Similar to the [knn-implementation](/documentation/2019/09/19/K-Nearest-Neighbor-Classifier/), the first step in this process is to load the raw CIFAR-10 data into python. A quick preview of the loaded data is shown in Figure 1 below.   

![CIFAR Sample](/assets/png/knn/cifar_10_sample.png){:width="560px"}  
__Figure 1:__ _Samples from the CIFAR-10 Dataset_
{: style="text-align: center;"} 


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

## <a name="pre-processing"></a> Data Pre-Processing

```python
Training data shape:  (49000, 3072)
Validation data shape:  (1000, 3072)
Test data shape:  (1000, 3072)
dev data shape:  (500, 3072)
```

![png](/assets/png/svm_files/svm_7_1.png)


## <a name="Linear-Classification"></a> Linear Classification




# <a name="Linear-Score-Function"></a> Linear Score Function

$$
f(x_{i}, W, b) = Wx_{i}+b
$$

# <a name="Loss-Function"></a> Loss Function


$$
L_{i} = \sum_{j \neq y_{i}} max(0,s_{j} - s_{y_{i}} + \Delta)
$$


# <a name="SVM"></a> Multiclass Support Vector Machine (SVM)

![png](/assets/png/svm_files/svm_17_0.png)

# <a name="SGD"></a> Stochastic Gradient Descent

![png](/assets/png/svm_files/svm_20_0.png)

# <a name="Hyperparameter-tuning"></a> Hyperparameter Tuning and Cross Validation

![png](/assets/png/svm_files/svm_22_0.png)
