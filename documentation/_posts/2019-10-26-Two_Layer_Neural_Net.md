---
layout: post
title: "Building a Two Layer Neural Network"
date: 2019-10-26
category: documentation
comments: true
math: true
author: "J. Abbott"
---

> Self Guided study of the [course notes](http://cs231n.github.io/)[^1] for cs231n: Convolutional Neural Networks for Visual Recognition provided through Stanford University. Included in this page are references to the classes and function calls developed by Stanford university and others, [Miranda[^2], Voss[^3]], who have worked through this course material. The complete implementation of the [Two-Layer Neural Network](/jupyter_notebook/jupyter%20notebooks/2019/09/26/svm_implementation.html) has been exported as a final Markdown file and can be found in the [Jupyter Notebooks](/jupyter_notebooks/) section of this site.

In this section, we'll work through the steps and underlying theory for developing a Two Layer Neural Network. The initial model will be generated and tested using a toy data set. Once we're able to obtain ideal testing and validation results using the toy dataset, we'll extend the application of the Two Layer Neural Network onto the CIFAR-10[^4] [^5] labeled dataset for image classification. 

- [Neural Network Architecture](#nn-architecture)
    - [Data Organization and Computation](#data_org)
- [Development (Toy Model)](#dev-toy)
    - [Weight Initialization](#w_init)
    - [Forward Pass Computation](#Forward-Pass-Computation)
        - [Activation Function](#activation-function)
        - [Loss Function](#loss-function)
        - [Regularization](#reg)
    - [SGD Through Backpropogation ](#backprop)
    - [Parameter Updates](#param-updates)
- [Development (CIFAR-10)](#dev-CIFAR)
- [Hyperparameter Tuning](#Hyperparameter-tuning)
- [Refereneces](#Ref)


## <a name="nn-architecture"></a> Neural Network Architecture

### <a name="data_org"></a> Data Organization and Computation
In the previous section covering an [SVM classifier](/documentation/2019/09/26/Training-an-SVM-Classifier.html), we developed an image classification model using a linear score function which was computed using the dot product of the input training data set $$X$$ and the randomly generated weight matrix $$ W $$. In this example, $$X$$ was a set of $$n$$ images, (i.e. row vectors) containing 3072 pixels each plus an additional bias dimension for a final shape of $$(n$$ x $$ 3073)$$. The weight matrix $$W$$ in this example consisted of 10 column vectors of size 3072 randomly generated weights, (plus a bias $$b$$). The final matrix of shape $$(3072$$ x $$10)$$ represented each possible classification within the data set, providing us with a set of parameters $$W$$ that could then be learned through Stochastic Gradient Descent, (SGD).  

For this implementation of a neural network, we'll utilize this same linear scoring function as a component within each layer of our network. The output from the first layer, also known as the hidden layer, will then be passed as an input into the second subsequent layer. As a general example, we might generate two weight matrices, $$W_1$$ of shape $$(3073 $$ x $$ 100)$$ and  $$W_2$$ of shape $$(10 $$ x $$ 100)$$, where the larger matrix $$W_1$$ will learn to identify larger features in the data set while the smaller weight matrix $$W_2$$ will learn to identify smaller, detail specific features. The final score is then computed as follows. 

$$
s = W_2 * max(0, X * W_1)
$$ 

In this representation, the use of the $$max()$$ function serves as a ReLU activation function to the network, (discussed below). This application provides us with a non-linearity which separates the two weight matrices, allowing us to train each set of parameters through SGD. This implementation of a neural network effectively executes a series of linear mapping functions which are tied together through activation functions, (i.e. non-linear functions). A simple visualization of this network is shown in Figure 1.

![png](/assets/png/2lnn/2LNN.png){:width="560px"}
{: style="text-align: center;"} 

__Figure 1:__ _Graphical Representation of a Two Layer Neural Network_
{: style="text-align: center;"} 

Note that the "first layer" of an N-Layer neural network starts after the input layer. In our case, figure one consists of an input layer and a 2 layer neural network, where by the network consists of a hidden layer and an output layer. Also omitted from the graphic in figure one are details regarding the activation function which is calculated as part of the hidden layer.  

## <a name="dev-toy"></a> Development (Toy Model)

Having a basic foundation for the architecture of the network, we begin development using a toy data set of arbitrary data $$X$$ and the associated labels $$y$$. Looking at the input data set $$X$$, we see that we have a total of 5, 4-dimensional data points in the $$(5$$ x $$4)$$ matrix.

```python
print(X)
[[ 16.24345364  -6.11756414  -5.28171752 -10.72968622]
 [  8.65407629 -23.01538697  17.44811764  -7.61206901]
 [  3.19039096  -2.49370375  14.62107937 -20.60140709]
 [ -3.22417204  -3.84054355  11.33769442 -10.99891267]
 [ -1.72428208  -8.77858418   0.42213747   5.82815214]]

print(y)
[0 1 2 2 1]
```

While creating this data set, we simultaneously generate an instance of our two-layer neural network. 
```python
net = TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)
```
Although we're currently developing a neural network to classify the "toy data set" noted above, the same class instance will be utilized below when applied to the CIFAR-10 dataset. As such, the input parameters to the class instance will be discussed with respect to both the toy data set and the CIFAR-10 data set. This will hopefully provide additional context to the class design, input parameters, and how the data sets relate to one another, specifically the input data $$X$$ and weight matrices $$W_1$$ and $$W_2$$. The input parameters for this class instance are defined as follows:

- input_size: Dimension D
    - Captures the second dimension of the training data set, (i.e. number of columns) 
    - Toy Data Set: D = 4
    - CIFAR-10 Set: D = 3072
- hidden_size:
    - Specifies the size of the second dimension within the first weight matrix $$W_1$$
        - Toy Data Set: 10
        - CIFAR-10 Set: Range between 50 and 300 (discussed below)
- num_classes:
    - Total number of classes in final output
        - Toy Data Set: 5
        - CIFAR-10 Set: 10




### <a name="w_init"></a> Weight Initialization

Using the input parameters defined above, the class instantiation generates weight and bias parameters internally. both weight matrices are initialized to small random values which are further scaled according to the input parameter $$std$$. Bias terms are initialized to zero, however these values are updated at a later step while training the model. Note that the random seed is set to a value of zero which produces the same set of random variables with every instance. This allows for repeatable testing and trouble shooting during development. 

```python
self.params = {}
np.random.seed(0)
self.params['W1'] = std * np.random.randn(input_size, hidden_size)
self.params['b1'] = np.zeros(hidden_size)
self.params['W2'] = std * np.random.randn(hidden_size, output_size)
self.params['b2'] = np.zeros(output_size)
```

### <a name="Forward-Pass-Computation"></a> Forward Pass Computation

The forward pass defines the steps in which we execute linear mapping, activation functions, apply regularization, and calculate the loss of the new scoring function. Conversely, during back propagation, (discussed below), we compute the gradient by propagating backwards through the forward pass computation using the chain rule. Using the toy data set $$X$$ defined above, the forward pass computation starts by calculating the linear score function of the input data set and the first weight matrix $$W_1$$. The activation function is then applied which thresholds all activations below zero to a value of zero. The results of this output are then passed into the next linear mapping function to generate the final score output for each class.

```python
# First layer linear score function
h1 = X.dot(W1) + b1

# First layer activation function
a1 = np.maximum(0, h1)

# Output Layer score function
scores = a1.dot(W2) + b2
```

# <a name="activation-function"></a> Activation Function

Using a biological neuron as a model, the activation function is used to model the firing rate of a neuron. In the biological model, each neuron receives input information from its dendrites and produces output information along the axon which is then carried to other neurons. We model the input information as a multiplication of the input data $$X$$ and the learned weights $$W$$ which represent the synaptic strength of the neuron. If the linear combination of this input information exceeds a certain threshold, than the neuron will "fire", sending and output signal along the axon to other neurons. We model this functionality using the Rectified Linear Unit, (ReLU), function which computes the function $$f(x) = max(0, x)$$. Similar to the biological neuron, the input provided to the ReLU function is the output from the linear scoring function $$s = xW$$, which corresponds to the input data and learned weights. From here the ReLU, (i.e. activation function) will than output the linear scoring function $$s$$ if the value of said score exceeds a certain threshold, (i.e. zero). Other commonly used activation functions include the sigmoid function, Tanh, ReLU, Leaky ReLU and Maxout, all of which are discussed in great detail under the [Neural Networks Part 1 - Commonly Used Activation Functions](http://cs231n.github.io/neural-networks-1/#actfun) section of the course notes. For our purposes, we'll stick with the Rectified Linear Unit, (ReLU). 


# <a name="loss-function"></a> Loss Function

Using the [Minimal Neural Network Case Study](http://cs231n.github.io/neural-networks-case-study/#loss)[^1] section from the course notes as a reference for computing the loss, we can implement the loss function for a Softmax classifier. The loss of the class scores using the loss function is defined as:

$$ 
L_i = -log( {e^{f_{y_i}} \over \sum_{j}e^{f_j}}) 
$$

As mentioned in the same section, "the full Softmax classifier loss is then defined as the average cross-entropy loss over the training examples and the regularization"[^1]:

$$
L = {1\over N} \sum_i{L_i} + {1\over 2} \lambda \sum_k \sum_l W_{k,l}^{2} 
$$


```python
exp_scores = np.exp(scores)
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

corect_logprobs = -np.log(probs[range(N), y])
data_loss = np.sum(corect_logprobs) / N
reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
loss = data_loss + reg_loss
```
# <a name="reg"></a> Regularization

As shown in the code snippet above, regularization is applied to the loss function to avoid over fitting. Over fitting occurs when the model learns to many of the details within the data set, including undesirable information such as noise. Regularization allows us to prevent over fitting by causing the network to generalize information affectively thus improving overall performance. In this section, we implement L2 regularization, which has the affect of penalizing the squared magnitude of the weight matrices $$W$$. L2 regularization is also known as weight decay since it has the effect of causing the weights to decay towards zero. Note that the regularization strength $$\lambda$$ is one of the hyperparameters we'll analyze at a later step as part of hyperparameter optimization.  

### <a name="backprop"></a> SGD Through Backpropogation

From the [Minimal Neural Network Case Study](http://cs231n.github.io/neural-networks-case-study/#loss)[^1] in the course notes, we're given the following expression along with the final evaluation for the gradient $${\partial L_i \over \partial f_k}$$.

$$
p_k = {e^{f_k} \over \sum_j e^{f_j}} 
$$

$$
L_i = -log(p_{y_i})
$$

$$
{\partial L_i \over \partial f_k} = (p_k-1)(y_i = k)
$$

```python
# compute gradient on scores
dscores = probs
dscores[range(N),y] -= 1
dscores /= N
```
With the gradient for the scores available, we can now backpropogate through the two layer network and calculate the gradient with respect to the weight and bias parameters. We can evaluate these expressions through dimension analysis and identification of various [Patterns in backward flow](http://cs231n.github.io/optimization-2/#patterns)[^1]. As described in the course notes, the local gradient of a multiply gate has the property of multiplying the input parameters against the backpropogated gradient received from the "downstream" function. For example, given the following expression,

$$
w * x = d
$$ 

And given the gradient of the end function $$f$$ with respect to $$d$$

$$
{\partial f \over \partial d}
$$

The $${\partial f \over \partial w}$$ through backpropogation is derived as:

$$ \begin{aligned}
{\partial f \over \partial w}   &= {\partial f \over \partial d}{\partial f \over \partial w} \\
                                &= {\partial f \over \partial d}{\partial \over \partial w}(w * x) \\
                                &= {\partial f \over \partial d} x
\end{aligned}$$

By intuition:

$$
{\partial f \over \partial x} = {\partial f \over \partial d} w
$$ 


Extending this analysis to [Gradients of vectorized operations](http://cs231n.github.io/optimization-2/#mat)[^1], we can apply the same logic and deduce the organization of the matrix multiplication by analysis of each matrices dimensions. The final implementation is given below.

```Python
# gradient for W2 and b2
grads['W2'] = np.dot(a1.T, dscores)
grads['b2'] = np.sum(dscores, axis=0)

# Backpropagate to h1
dh1 = np.dot(dscores, W2.T)

# Backpropagate ReLU non-linearity
dh1[a1 <= 0] = 0

# 
grads['W1'] = np.dot(X.T, dh1)
grads['b1'] = np.sum(dh1, axis=0)
                
grads['W2'] += 2 * reg * W2
grads['W1'] += 2 * reg * W1
```

### <a name="param-updates"></a> Parameter Updates

Having calculated the gradients, we can now perform a parameter optimization. For this implementation, we'll stick with the "vanilla update" which updates the parameters along the negative gradient direction and scales the update according to the learning rate constant. 

```python
self.params['W1'] += -learning_rate * grads['W1']
self.params['b1'] += -learning_rate * grads['b1']
self.params['W2'] += -learning_rate * grads['W2']
self.params['b2'] += -learning_rate * grads['b2']  
```

## <a name="train"></a> Training

Finally, we want to train and evaluate the performance of our two-layer neural network. We'll first create an instance of the class object TwoLayerNet() and then provide the class instance with the toy data set and the hyperparameters "learning rate" and "regularization strength".  

```python
net = init_toy_model()
stats = net.train(X, y, X, y,
            learning_rate=1e-1, reg=5e-6,
            num_iters=100, verbose=False)
```

Without speaking to the provided code in the train() function, the first task is to generate a mini batch of the training data. One of the reasons for doing this step is to avoid degradation of the model quality. The three contrasting types of gradient descent are Stochastic Gradient Descent, Batch Gradient Descent, and Mini-Batch Gradient Descent, (each of which having their own pros and cons). Stochastic gradient Descent updates the parameters after each example within the training data set. Batch Gradient Descent calculates the error for all training examples, (i.e. one training epoch) before updating the parameters. Mini-batch Gradient Descent attempts to combine the two methodologies by lumping the training data into mini-batches. Updates to the parameters are made after evaluating the error for each example within the mini-batch. 

```python
idx = np.random.choice(range(X.shape[0]), batch_size)
X_batch = X[idx]
y_batch = y[idx]
```

Sticking with the mini-batch implementation as noted by the assignment instructions, we then calculate the loss and gradient for the current batch and update the parameters as follows. 

```python
self.params['W1'] += -learning_rate * grads['W1']
self.params['b1'] += -learning_rate * grads['b1']
self.params['W2'] += -learning_rate * grads['W2']
self.params['b2'] += -learning_rate * grads['b2']   
```

![png](/assets/png/2lnn/output_13_1.png){:width="560px"}
{: style="text-align: center;"} 

__Figure 2:__ _Training loss after each training iteration_
{: style="text-align: center;"} 

Plotting these results, we can observe a decreasing training loss which converges around 0.017. From this, we can conclude that everything is working as expected. We can now begin test this implementation against the larger CIRFAR-10 data set. 

## <a name="dev-CIFAR"></a> Development (CIFAR-10)

Similar to the [knn-implementation](/documentation/2019/09/19/K-Nearest-Neighbor-Classifier.html) and SVM-Classifier, the first step in this process is to load the raw CIFAR-10 data into python. A quick preview of the loaded data is shown in Figure 4 below.  

![png](/assets/png/2lnn/output_15_0.png){:width="560px"}
{: style="text-align: center;"} 

__Figure 4:__ _Samples from the CIFAR-10 Dataset_
{: style="text-align: center;"} 

After data preprocessing, which includes zero averaging the data set and organizing each image into a single column vector, we end up with the following data sets. For clarity, the shape of each data set is noted. 

```python
X_train: Train data shape:  (49000, 3072)
y_train: Train labels shape:  (49000,)
X_val: Validation data shape:  (1000, 3072)
y_val: Validation labels shape:  (1000,)
X_test: Test data shape:  (1000, 3072)
y_test: Test labels shape:  (1000,)
```


```python
# Normalize the data: subtract the mean image
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
```

With minimal effort, we can create a neural net class instance and train the network using the CIFAR-10 data set. Note that the train function calls the loss function internal which calculates the loss and gradients discussed above. 

```python
input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)
print(net.params['W1'].shape)
# Train the network
stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=1000, batch_size=200,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.25, verbose=True)
```
Setting the verbose parameter equal to True, we can view the convergence of the loss function as we train the network. From this simple output, we see that the results are not favorable, converging at a value around 1.967. Moreover, testing the trained network against the validation data set, we receive an accuracy of 28%. Comparing these results to that of the [SVM](http://localhost:4000/documentation/2019/09/26/Training-an-SVM-Classifier.html#Hyperparameter-tuning) from the previous section, where we obtained an accuracy of 38.9%, we can conclude that the network is not performing optimally. This then leads to the task of debugging and hyper parameter tuning.  

```python
iteration 0 / 1000: loss 2.302976
iteration 100 / 1000: loss 2.302638
iteration 200 / 1000: loss 2.299256
iteration 300 / 1000: loss 2.270635
iteration 400 / 1000: loss 2.219117
iteration 500 / 1000: loss 2.141064
iteration 600 / 1000: loss 2.129066
iteration 700 / 1000: loss 2.047454
iteration 800 / 1000: loss 1.988162
iteration 900 / 1000: loss 1.967131

# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)

Validation accuracy:  0.282
```

## <a name="Hyperparameter-tuning"></a> Hyperparameter Tuning


![png](/assets/png/2lnn/output_19_0.png){:width="560px"}
{: style="text-align: center;"} 

__Figure 5:__ _..._
{: style="text-align: center;"} 


![png](/assets/png/2lnn/output_21_0.png){:width="560px"}
{: style="text-align: center;"} 

__Figure 6:__ _..._
{: style="text-align: center;"} 


![png](/assets/png/2lnn/output_26_0.png){:width="560px"}
{: style="text-align: center;"} 

__Figure 7:__ _..._
{: style="text-align: center;"} 




# <a name="Ref"></a> References

[^1]: CS231n Stanford University (2015, Aug).Convolutional Neural Networks for Visual Recognition [Web log post]. Retrieved from [http://cs231n.stanford.edu/](http://cs231n.stanford.edu/)

[^2]: Miranda, L. J. (2017, Feb 17). Implementing a two-layer neural network from scratch [Web log post]. Retrieved from [https://ljvmiranda921.github.io/notebook/2017/02/17/artificial-neural-networks/](https://ljvmiranda921.github.io/notebook/2017/02/17/artificial-neural-networks/)

[^3]: Voss, C. (2015, Sep 22).  CNN-Assignments [Web log post]. Retrieved from [https://github.com/CatalinVoss/cnn-assignments/tree/master/assignment1](https://github.com/CatalinVoss/cnn-assignments/tree/master/assignment1)

[^4]: Krizhevsky, A., Nair, V., and Hinton, G. (2009). CIFAR-10 (Canadian Institute for Advanced Research) [Web log post]. Retrieved from [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

[^5]: Krizhevsky, A., 2009. [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

