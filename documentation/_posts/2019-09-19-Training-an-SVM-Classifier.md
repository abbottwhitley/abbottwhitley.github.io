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

In this section, we continue the task of image classification on the CIFAR-10 labeled dataset. In this approach we'll implement the Support Vector Machine, (SVM), a linear classification, supervised learning algorithm that works to identify the optimal hyperplane for linearly separable patterns.The final output of the model is a class identity. The SVM uses a linear function which assumes a boundary exists that separates one class boundary from another. The primary goal of the SVM is to efficiently find the boundary which separates one class from another. In this implementation, we will utilize a linear score function to compute a class score for the input data set. The output score can then be used within a loss function to better determine the success of the linear score function. Stochastic Gradient Descent will then be utilized as the optimization algorithm to minimize the loss obtained by the loss function. 

- [Loading the CIFAR-10 dataset](#loading-the-cifar-10-dataset)
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
Prior to training the model, the data must be partitioning accordingly. From the 50,000 images within the training data, 49,000 images will be grouped as the official training set while the remaining 1,000 images will be designated as a validation set. The validation set will be used to tune the learning rate and regularization strength. From the 10,000 images within the test data set, a subsample of 1,000 images will be used to evaluate the accuracy of the SVM. A separate development data set of 500 randomly selected images will be created for use during development. Also note that the pixels for each image have been reshaped from a 3-dimensional, (32 x 32) matrix into a 3072 element array.  

```python
Training data shape:  (49000, 3072)
Validation data shape:  (1000, 3072)
Test data shape:  (1000, 3072)
dev data shape:  (500, 3072)
```

Finally, we'll normalize the dataset by subtracting the mean of the training data from each of the datasets outlined above. Subtracting the dataset mean centers the data and helps to keep the feature dataset within a similar range of each other which is beneficial when processing the gradient.  

![CIFAR Sample](/assets/png/svm_files/svm_7_1.png){:width="300px"}  
__Figure 2:__ _Mean of the Training Data Set_
{: style="text-align: center;"} 


## <a name="Linear-Classification"></a> Linear Classification

This method of linear classification is in effect a template matching algorithm where each class within an input weight matrix $$W$$ is iteratively adjusted to create a best fit template for the class it represents. The adjusted weights can then be used to make a prediction on an input test image. This is accomplished through a single dot product of the two matrices, (the test image $$x_{i}$$ and the adjusted weight matrix $$W$$). The output of this process will then be a final score for each class which specifies how closely the input image maps to the corresponding class.   

# <a name="Linear-Score-Function"></a> Linear Score Function

The SVM is initiated by first computing the linear score function of the data set. The linear score function is a product of the input training data, $$x_{i}$$, the randomly generated weight matrix $$ W $$, and a bias vector $$ B $$ which influences the output score without directly interacting with the input training data. 
 
```python
# generate a random SVM weight matrix of small numbers
W = np.random.randn(3073, 10) * 0.0001 
``` 

From these parameters, we arrive at the following function. 

$$
f(x_{i}, W, b) = Wx_{i}+b
$$


A few observations can be made about the linear score function noted above. The dot product of the two matrices $$Wx_{i}$$ result in an array of 10 separate scores for each image. Noting here that the weights $$W$$ were randomly generated indicates that we have the ability to adjust these input values. As we train the model, the goal is to adjust the output to a significantly higher score for the correct class, with respect to the output scores for the incorrect classes. This step of fine-tunning the model will be accomplished with stochastic gradient descent. 

# <a name="Loss-Function"></a> Loss Function


$$
L_{i} = \sum_{j \neq y_{i}} max(0,s_{j} - s_{y_{i}} + \Delta)
$$


# <a name="SVM"></a> Multiclass Support Vector Machine (SVM)

![png](/assets/png/svm_files/svm_17_0.png)

# <a name="SGD"></a> Stochastic Gradient Descent

...SVM works to increase the margin between the identified patterns. 

![png](/assets/png/svm_files/svm_20_0.png)

# <a name="Hyperparameter-tuning"></a> Hyperparameter Tuning and Cross Validation

![png](/assets/png/svm_files/svm_22_0.png)


Conclusion (Benifits)
	- SVM uses training data to learn the parameters of W, and B. Training data is then discarded and the parameters are used to make predictions
	- Clasifying a test image involves a single matrix multiplication and addition, less expensive from a computation perspective
	- Discussion around how different color channels for specific objects might result in specific weights for those channels
	- Template matching