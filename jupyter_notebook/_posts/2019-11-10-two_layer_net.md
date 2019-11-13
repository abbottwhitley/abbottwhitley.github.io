# Implementing a Neural Network
In this exercise we will develop a neural network with fully-connected layers to perform classification, and test it out on the CIFAR-10 dataset.


```python
# A bit of setup

import numpy as np
import matplotlib.pyplot as plt

from cs231n.classifiers.neural_net import TwoLayerNet

%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
```

We will use the class `TwoLayerNet` in the file `cs231n/classifiers/neural_net.py` to represent instances of our network. The network parameters are stored in the instance variable `self.params` where keys are string parameter names and values are numpy arrays. Below, we initialize toy data and a toy model that we will use to develop your implementation.


```python
# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y

net = init_toy_model()
X, y = init_toy_data()
```


```python
print(X)
```

    [[ 16.24345364  -6.11756414  -5.28171752 -10.72968622]
     [  8.65407629 -23.01538697  17.44811764  -7.61206901]
     [  3.19039096  -2.49370375  14.62107937 -20.60140709]
     [ -3.22417204  -3.84054355  11.33769442 -10.99891267]
     [ -1.72428208  -8.77858418   0.42213747   5.82815214]]
    


```python
print(y)
```

    [0 1 2 2 1]
    

# Forward pass: compute scores
Open the file `cs231n/classifiers/neural_net.py` and look at the method `TwoLayerNet.loss`. This function is very similar to the loss functions you have written for the SVM and Softmax exercises: It takes the data and weights and computes the class scores, the loss, and the gradients on the parameters. 

Implement the first part of the forward pass which uses the weights and biases to compute the scores for all inputs.


```python
scores = net.loss(X)
print('Your scores:')
print(scores)
print()
print('correct scores:')
correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print(correct_scores)
print()

# The difference should be very small. We get < 1e-7
print('Difference between your scores and correct scores:')
print(np.sum(np.abs(scores - correct_scores)))
```

    Your scores:
    [[-0.81233741 -1.27654624 -0.70335995]
     [-0.17129677 -1.18803311 -0.47310444]
     [-0.51590475 -1.01354314 -0.8504215 ]
     [-0.15419291 -0.48629638 -0.52901952]
     [-0.00618733 -0.12435261 -0.15226949]]
    
    correct scores:
    [[-0.81233741 -1.27654624 -0.70335995]
     [-0.17129677 -1.18803311 -0.47310444]
     [-0.51590475 -1.01354314 -0.8504215 ]
     [-0.15419291 -0.48629638 -0.52901952]
     [-0.00618733 -0.12435261 -0.15226949]]
    
    Difference between your scores and correct scores:
    3.6802720496109664e-08
    

# Forward pass: compute loss
In the same function, implement the second part that computes the data and regularization loss.


```python
loss, _ = net.loss(X, y, reg=0.05)
correct_loss = 1.30378789133

# should be very small, we get < 1e-12
print('Difference between your loss and correct loss:')
print(np.sum(np.abs(loss - correct_loss)))
```

    Difference between your loss and correct loss:
    1.794120407794253e-13
    

# Backward pass
Implement the rest of the function. This will compute the gradient of the loss with respect to the variables `W1`, `b1`, `W2`, and `b2`. Now that you (hopefully!) have a correctly implemented forward pass, you can debug your backward pass using a numeric gradient check:


```python
from cs231n.gradient_check import eval_numerical_gradient

# Use numeric gradient checking to check your implementation of the backward pass.
# If your implementation is correct, the difference between the numeric and
# analytic gradients should be less than 1e-8 for each of W1, W2, b1, and b2.

loss, grads = net.loss(X, y, reg=0.05)

# these should all be less than 1e-8 or so
for param_name in grads:
    f = lambda W: net.loss(X, y, reg=0.05)[0]
    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))
```

    W2 max relative error: 3.440708e-09
    b2 max relative error: 3.865028e-11
    W1 max relative error: 3.669858e-09
    b1 max relative error: 2.738422e-09
    

# Train the network
To train the network we will use stochastic gradient descent (SGD), similar to the SVM and Softmax classifiers. Look at the function `TwoLayerNet.train` and fill in the missing sections to implement the training procedure. This should be very similar to the training procedure you used for the SVM and Softmax classifiers. You will also have to implement `TwoLayerNet.predict`, as the training process periodically performs prediction to keep track of accuracy over time while the network trains.

Once you have implemented the method, run the code below to train a two-layer network on toy data. You should achieve a training loss less than 0.02.


```python
net = init_toy_model()
stats = net.train(X, y, X, y,
            learning_rate=1e-1, reg=5e-6,
            num_iters=100, verbose=True)

print('Final training loss: ', stats['loss_history'][-1])

# plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()
```

    iteration 0 / 100: loss 1.241994
    Final training loss:  0.017149607938732023
    


![png](output_13_1.png)


# Load the data
Now that you have implemented a two-layer network that passes gradient checks and works on toy data, it's time to load up our favorite CIFAR-10 data so we can use it to train a classifier on a real dataset.


```python
from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    
    # Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
    try:
       del X_train, y_train
       del X_test, y_test
       print('Clear previously loaded data.')
    except:
       pass

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    
    # Visualize some examples from the dataset.
    # We show a few examples of training images from each class.
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()
        
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test

# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
```


![png](output_15_0.png)


    Train data shape:  (49000, 3072)
    Train labels shape:  (49000,)
    Validation data shape:  (1000, 3072)
    Validation labels shape:  (1000,)
    Test data shape:  (1000, 3072)
    Test labels shape:  (1000,)
    

# Train a network
To train our network we will use SGD. In addition, we will adjust the learning rate with an exponential learning rate schedule as optimization proceeds; after each epoch, we will reduce the learning rate by multiplying it by a decay rate.


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

# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)

```

    (3072, 50)
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
    Validation accuracy:  0.282
    

# Debug the training
With the default parameters we provided above, you should get a validation accuracy of about 0.29 on the validation set. This isn't very good.

One strategy for getting insight into what's wrong is to plot the loss function and the accuracies on the training and validation sets during optimization.

Another strategy is to visualize the weights that were learned in the first layer of the network. In most neural networks trained on visual data, the first layer weights typically show some visible structure when visualized.


```python
# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.legend()
plt.show()
```


![png](output_19_0.png)



```python
print(stats['train_acc_history'])
print(stats['val_acc_history'])
```

    [0.125, 0.135, 0.175, 0.27, 0.31]
    [0.107, 0.162, 0.198, 0.25, 0.281]
    


```python
from cs231n.vis_utils import visualize_grid

# Visualize the weights of the network

def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()

show_net_weights(net)
```


![png](output_21_0.png)


# Tune your hyperparameters

**What's wrong?**. Looking at the visualizations above, we see that the loss is decreasing more or less linearly, which seems to suggest that the learning rate may be too low. Moreover, there is no gap between the training and validation accuracy, suggesting that the model we used has low capacity, and that we should increase its size. On the other hand, with a very large model we would expect to see more overfitting, which would manifest itself as a very large gap between the training and validation accuracy.

**Tuning**. Tuning the hyperparameters and developing intuition for how they affect the final performance is a large part of using Neural Networks, so we want you to get a lot of practice. Below, you should experiment with different values of the various hyperparameters, including hidden layer size, learning rate, numer of training epochs, and regularization strength. You might also consider tuning the learning rate decay, but you should be able to get good performance using the default value.

**Approximate results**. You should be aim to achieve a classification accuracy of greater than 48% on the validation set. Our best network gets over 52% on the validation set.

**Experiment**: You goal in this exercise is to get as good of a result on CIFAR-10 as you can (52% could serve as a reference), with a fully-connected Neural Network. Feel free implement your own techniques (e.g. PCA to reduce dimensionality, or adding dropout, or adding features to the solver, etc.).

**Explain your hyperparameter tuning process below.**

$\color{blue}{\textit Your Answer:}$


```python
best_net = None # store the best model into this 

#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

best_val = -1
best_stats = None
h = [100, 150, 200, 250]
learning_rates = [1e-3, 1e-4, 1e-5]
regularization_strengths = [0.3, 0.4, 0.5]
results = {}
iters = 1000
for hidden_size in h:
    for lr in learning_rates:
        for rs in regularization_strengths:
            print("Processing H: %s ::: Learning Rate: %s ::: Reg Strength: %s" % (hidden_size, lr, rs))
            net = TwoLayerNet(input_size, hidden_size, num_classes)

            # Train the network
            stats = net.train(X_train, y_train, X_val, y_val,
                        num_iters=iters, batch_size=200,
                        learning_rate=lr, learning_rate_decay=0.95,
                        reg=rs, verbose=True)

            y_train_pred = net.predict(X_train)
            acc_train = np.mean(y_train == y_train_pred)
            y_val_pred = net.predict(X_val)
            acc_val = np.mean(y_val == y_val_pred)

            results[(hidden_size, lr, rs)] = (acc_train, acc_val)

            if best_val < acc_val:
                best_stats = stats
                best_val = acc_val
                best_net = net

```

    Processing H: 100 ::: Learning Rate: 0.001 ::: Reg Strength: 0.3
    iteration 0 / 1000: loss 2.303496
    iteration 100 / 1000: loss 1.934023
    iteration 200 / 1000: loss 1.724827
    iteration 300 / 1000: loss 1.633215
    iteration 400 / 1000: loss 1.658637
    iteration 500 / 1000: loss 1.678121
    iteration 600 / 1000: loss 1.494559
    iteration 700 / 1000: loss 1.566209
    iteration 800 / 1000: loss 1.470060
    iteration 900 / 1000: loss 1.614719
    Processing H: 100 ::: Learning Rate: 0.001 ::: Reg Strength: 0.4
    iteration 0 / 1000: loss 2.303804
    iteration 100 / 1000: loss 1.937946
    iteration 200 / 1000: loss 1.734066
    iteration 300 / 1000: loss 1.646694
    iteration 400 / 1000: loss 1.671811
    iteration 500 / 1000: loss 1.688113
    iteration 600 / 1000: loss 1.514092
    iteration 700 / 1000: loss 1.588273
    iteration 800 / 1000: loss 1.486926
    iteration 900 / 1000: loss 1.632204
    Processing H: 100 ::: Learning Rate: 0.001 ::: Reg Strength: 0.5
    iteration 0 / 1000: loss 2.304112
    iteration 100 / 1000: loss 1.941509
    iteration 200 / 1000: loss 1.740702
    iteration 300 / 1000: loss 1.657684
    iteration 400 / 1000: loss 1.683584
    iteration 500 / 1000: loss 1.706019
    iteration 600 / 1000: loss 1.532358
    iteration 700 / 1000: loss 1.613551
    iteration 800 / 1000: loss 1.505119
    iteration 900 / 1000: loss 1.655387
    Processing H: 100 ::: Learning Rate: 0.0001 ::: Reg Strength: 0.3
    iteration 0 / 1000: loss 2.303496
    iteration 100 / 1000: loss 2.302578
    iteration 200 / 1000: loss 2.293809
    iteration 300 / 1000: loss 2.240220
    iteration 400 / 1000: loss 2.194275
    iteration 500 / 1000: loss 2.141821
    iteration 600 / 1000: loss 2.076900
    iteration 700 / 1000: loss 2.095224
    iteration 800 / 1000: loss 1.957757
    iteration 900 / 1000: loss 1.997002
    Processing H: 100 ::: Learning Rate: 0.0001 ::: Reg Strength: 0.4
    iteration 0 / 1000: loss 2.303804
    iteration 100 / 1000: loss 2.302882
    iteration 200 / 1000: loss 2.294183
    iteration 300 / 1000: loss 2.240999
    iteration 400 / 1000: loss 2.195382
    iteration 500 / 1000: loss 2.143304
    iteration 600 / 1000: loss 2.078640
    iteration 700 / 1000: loss 2.097250
    iteration 800 / 1000: loss 1.960251
    iteration 900 / 1000: loss 2.000153
    Processing H: 100 ::: Learning Rate: 0.0001 ::: Reg Strength: 0.5
    iteration 0 / 1000: loss 2.304112
    iteration 100 / 1000: loss 2.303183
    iteration 200 / 1000: loss 2.294552
    iteration 300 / 1000: loss 2.241769
    iteration 400 / 1000: loss 2.196474
    iteration 500 / 1000: loss 2.144774
    iteration 600 / 1000: loss 2.080330
    iteration 700 / 1000: loss 2.099245
    iteration 800 / 1000: loss 1.962711
    iteration 900 / 1000: loss 2.003229
    Processing H: 100 ::: Learning Rate: 1e-05 ::: Reg Strength: 0.3
    iteration 0 / 1000: loss 2.303496
    iteration 100 / 1000: loss 2.303451
    iteration 200 / 1000: loss 2.303409
    iteration 300 / 1000: loss 2.303364
    iteration 400 / 1000: loss 2.303343
    iteration 500 / 1000: loss 2.303288
    iteration 600 / 1000: loss 2.303191
    iteration 700 / 1000: loss 2.303214
    iteration 800 / 1000: loss 2.302990
    iteration 900 / 1000: loss 2.302996
    Processing H: 100 ::: Learning Rate: 1e-05 ::: Reg Strength: 0.4
    iteration 0 / 1000: loss 2.303804
    iteration 100 / 1000: loss 2.303757
    iteration 200 / 1000: loss 2.303715
    iteration 300 / 1000: loss 2.303670
    iteration 400 / 1000: loss 2.303647
    iteration 500 / 1000: loss 2.303592
    iteration 600 / 1000: loss 2.303495
    iteration 700 / 1000: loss 2.303517
    iteration 800 / 1000: loss 2.303293
    iteration 900 / 1000: loss 2.303299
    Processing H: 100 ::: Learning Rate: 1e-05 ::: Reg Strength: 0.5
    iteration 0 / 1000: loss 2.304112
    iteration 100 / 1000: loss 2.304064
    iteration 200 / 1000: loss 2.304020
    iteration 300 / 1000: loss 2.303974
    iteration 400 / 1000: loss 2.303951
    iteration 500 / 1000: loss 2.303895
    iteration 600 / 1000: loss 2.303798
    iteration 700 / 1000: loss 2.303819
    iteration 800 / 1000: loss 2.303595
    iteration 900 / 1000: loss 2.303601
    Processing H: 150 ::: Learning Rate: 0.001 ::: Reg Strength: 0.3
    iteration 0 / 1000: loss 2.304006
    iteration 100 / 1000: loss 1.874272
    iteration 200 / 1000: loss 1.741396
    iteration 300 / 1000: loss 1.679444
    iteration 400 / 1000: loss 1.592295
    iteration 500 / 1000: loss 1.630126
    iteration 600 / 1000: loss 1.657525
    iteration 700 / 1000: loss 1.564650
    iteration 800 / 1000: loss 1.533646
    iteration 900 / 1000: loss 1.496230
    Processing H: 150 ::: Learning Rate: 0.001 ::: Reg Strength: 0.4
    iteration 0 / 1000: loss 2.304467
    iteration 100 / 1000: loss 1.878552
    iteration 200 / 1000: loss 1.749327
    iteration 300 / 1000: loss 1.685700
    iteration 400 / 1000: loss 1.607196
    iteration 500 / 1000: loss 1.646800
    iteration 600 / 1000: loss 1.678160
    iteration 700 / 1000: loss 1.589396
    iteration 800 / 1000: loss 1.559482
    iteration 900 / 1000: loss 1.525471
    Processing H: 150 ::: Learning Rate: 0.001 ::: Reg Strength: 0.5
    iteration 0 / 1000: loss 2.304928
    iteration 100 / 1000: loss 1.882243
    iteration 200 / 1000: loss 1.756697
    iteration 300 / 1000: loss 1.695455
    iteration 400 / 1000: loss 1.616714
    iteration 500 / 1000: loss 1.664345
    iteration 600 / 1000: loss 1.699235
    iteration 700 / 1000: loss 1.606009
    iteration 800 / 1000: loss 1.573412
    iteration 900 / 1000: loss 1.541338
    Processing H: 150 ::: Learning Rate: 0.0001 ::: Reg Strength: 0.3
    iteration 0 / 1000: loss 2.304006
    iteration 100 / 1000: loss 2.302461
    iteration 200 / 1000: loss 2.285757
    iteration 300 / 1000: loss 2.205187
    iteration 400 / 1000: loss 2.159101
    iteration 500 / 1000: loss 2.148959
    iteration 600 / 1000: loss 2.067656
    iteration 700 / 1000: loss 2.022951
    iteration 800 / 1000: loss 1.962501
    iteration 900 / 1000: loss 1.929666
    Processing H: 150 ::: Learning Rate: 0.0001 ::: Reg Strength: 0.4
    iteration 0 / 1000: loss 2.304467
    iteration 100 / 1000: loss 2.302918
    iteration 200 / 1000: loss 2.286339
    iteration 300 / 1000: loss 2.206267
    iteration 400 / 1000: loss 2.160506
    iteration 500 / 1000: loss 2.150289
    iteration 600 / 1000: loss 2.069575
    iteration 700 / 1000: loss 2.025490
    iteration 800 / 1000: loss 1.965476
    iteration 900 / 1000: loss 1.932975
    Processing H: 150 ::: Learning Rate: 0.0001 ::: Reg Strength: 0.5
    iteration 0 / 1000: loss 2.304928
    iteration 100 / 1000: loss 2.303371
    iteration 200 / 1000: loss 2.286913
    iteration 300 / 1000: loss 2.207335
    iteration 400 / 1000: loss 2.161898
    iteration 500 / 1000: loss 2.151611
    iteration 600 / 1000: loss 2.071493
    iteration 700 / 1000: loss 2.028024
    iteration 800 / 1000: loss 1.968436
    iteration 900 / 1000: loss 1.936315
    Processing H: 150 ::: Learning Rate: 1e-05 ::: Reg Strength: 0.3
    iteration 0 / 1000: loss 2.304006
    iteration 100 / 1000: loss 2.303910
    iteration 200 / 1000: loss 2.303794
    iteration 300 / 1000: loss 2.303709
    iteration 400 / 1000: loss 2.303638
    iteration 500 / 1000: loss 2.303646
    iteration 600 / 1000: loss 2.303545
    iteration 700 / 1000: loss 2.303377
    iteration 800 / 1000: loss 2.303087
    iteration 900 / 1000: loss 2.302589
    Processing H: 150 ::: Learning Rate: 1e-05 ::: Reg Strength: 0.4
    iteration 0 / 1000: loss 2.304467
    iteration 100 / 1000: loss 2.304370
    iteration 200 / 1000: loss 2.304253
    iteration 300 / 1000: loss 2.304167
    iteration 400 / 1000: loss 2.304095
    iteration 500 / 1000: loss 2.304102
    iteration 600 / 1000: loss 2.304000
    iteration 700 / 1000: loss 2.303832
    iteration 800 / 1000: loss 2.303543
    iteration 900 / 1000: loss 2.303046
    Processing H: 150 ::: Learning Rate: 1e-05 ::: Reg Strength: 0.5
    iteration 0 / 1000: loss 2.304928
    iteration 100 / 1000: loss 2.304829
    iteration 200 / 1000: loss 2.304711
    iteration 300 / 1000: loss 2.304624
    iteration 400 / 1000: loss 2.304551
    iteration 500 / 1000: loss 2.304556
    iteration 600 / 1000: loss 2.304454
    iteration 700 / 1000: loss 2.304285
    iteration 800 / 1000: loss 2.303995
    iteration 900 / 1000: loss 2.303499
    Processing H: 200 ::: Learning Rate: 0.001 ::: Reg Strength: 0.3
    iteration 0 / 1000: loss 2.304445
    iteration 100 / 1000: loss 1.849661
    iteration 200 / 1000: loss 1.745126
    iteration 300 / 1000: loss 1.816652
    iteration 400 / 1000: loss 1.673047
    iteration 500 / 1000: loss 1.542567
    iteration 600 / 1000: loss 1.585752
    iteration 700 / 1000: loss 1.588187
    iteration 800 / 1000: loss 1.463372
    iteration 900 / 1000: loss 1.614004
    Processing H: 200 ::: Learning Rate: 0.001 ::: Reg Strength: 0.4
    iteration 0 / 1000: loss 2.305060
    iteration 100 / 1000: loss 1.854159
    iteration 200 / 1000: loss 1.752807
    iteration 300 / 1000: loss 1.828637
    iteration 400 / 1000: loss 1.688073
    iteration 500 / 1000: loss 1.565137
    iteration 600 / 1000: loss 1.610989
    iteration 700 / 1000: loss 1.607205
    iteration 800 / 1000: loss 1.488921
    iteration 900 / 1000: loss 1.634788
    Processing H: 200 ::: Learning Rate: 0.001 ::: Reg Strength: 0.5
    iteration 0 / 1000: loss 2.305676
    iteration 100 / 1000: loss 1.857541
    iteration 200 / 1000: loss 1.760619
    iteration 300 / 1000: loss 1.838576
    iteration 400 / 1000: loss 1.696546
    iteration 500 / 1000: loss 1.580356
    iteration 600 / 1000: loss 1.627308
    iteration 700 / 1000: loss 1.627342
    iteration 800 / 1000: loss 1.504788
    iteration 900 / 1000: loss 1.661733
    Processing H: 200 ::: Learning Rate: 0.0001 ::: Reg Strength: 0.3
    iteration 0 / 1000: loss 2.304445
    iteration 100 / 1000: loss 2.302720
    iteration 200 / 1000: loss 2.287593
    iteration 300 / 1000: loss 2.243041
    iteration 400 / 1000: loss 2.187324
    iteration 500 / 1000: loss 2.051106
    iteration 600 / 1000: loss 2.088828
    iteration 700 / 1000: loss 2.039649
    iteration 800 / 1000: loss 1.880027
    iteration 900 / 1000: loss 2.021152
    Processing H: 200 ::: Learning Rate: 0.0001 ::: Reg Strength: 0.4
    iteration 0 / 1000: loss 2.305060
    iteration 100 / 1000: loss 2.303328
    iteration 200 / 1000: loss 2.288316
    iteration 300 / 1000: loss 2.244057
    iteration 400 / 1000: loss 2.188879
    iteration 500 / 1000: loss 2.053080
    iteration 600 / 1000: loss 2.091002
    iteration 700 / 1000: loss 2.042297
    iteration 800 / 1000: loss 1.883521
    iteration 900 / 1000: loss 2.024723
    Processing H: 200 ::: Learning Rate: 0.0001 ::: Reg Strength: 0.5
    iteration 0 / 1000: loss 2.305676
    iteration 100 / 1000: loss 2.303932
    iteration 200 / 1000: loss 2.289030
    iteration 300 / 1000: loss 2.245058
    iteration 400 / 1000: loss 2.190406
    iteration 500 / 1000: loss 2.055041
    iteration 600 / 1000: loss 2.093133
    iteration 700 / 1000: loss 2.044913
    iteration 800 / 1000: loss 1.886935
    iteration 900 / 1000: loss 2.028241
    Processing H: 200 ::: Learning Rate: 1e-05 ::: Reg Strength: 0.3
    iteration 0 / 1000: loss 2.304445
    iteration 100 / 1000: loss 2.304378
    iteration 200 / 1000: loss 2.304266
    iteration 300 / 1000: loss 2.304223
    iteration 400 / 1000: loss 2.304078
    iteration 500 / 1000: loss 2.303868
    iteration 600 / 1000: loss 2.303887
    iteration 700 / 1000: loss 2.303653
    iteration 800 / 1000: loss 2.303228
    iteration 900 / 1000: loss 2.303659
    Processing H: 200 ::: Learning Rate: 1e-05 ::: Reg Strength: 0.4
    iteration 0 / 1000: loss 2.305060
    iteration 100 / 1000: loss 2.304992
    iteration 200 / 1000: loss 2.304878
    iteration 300 / 1000: loss 2.304834
    iteration 400 / 1000: loss 2.304688
    iteration 500 / 1000: loss 2.304477
    iteration 600 / 1000: loss 2.304495
    iteration 700 / 1000: loss 2.304261
    iteration 800 / 1000: loss 2.303836
    iteration 900 / 1000: loss 2.304265
    Processing H: 200 ::: Learning Rate: 1e-05 ::: Reg Strength: 0.5
    iteration 0 / 1000: loss 2.305676
    iteration 100 / 1000: loss 2.305606
    iteration 200 / 1000: loss 2.305490
    iteration 300 / 1000: loss 2.305444
    iteration 400 / 1000: loss 2.305296
    iteration 500 / 1000: loss 2.305084
    iteration 600 / 1000: loss 2.305100
    iteration 700 / 1000: loss 2.304865
    iteration 800 / 1000: loss 2.304440
    iteration 900 / 1000: loss 2.304867
    Processing H: 250 ::: Learning Rate: 0.001 ::: Reg Strength: 0.3
    iteration 0 / 1000: loss 2.304864
    iteration 100 / 1000: loss 1.858971
    iteration 200 / 1000: loss 1.777485
    iteration 300 / 1000: loss 1.589951
    iteration 400 / 1000: loss 1.588001
    iteration 500 / 1000: loss 1.695929
    iteration 600 / 1000: loss 1.544592
    iteration 700 / 1000: loss 1.728002
    iteration 800 / 1000: loss 1.445239
    iteration 900 / 1000: loss 1.615519
    Processing H: 250 ::: Learning Rate: 0.001 ::: Reg Strength: 0.4
    iteration 0 / 1000: loss 2.305633
    iteration 100 / 1000: loss 1.863586
    iteration 200 / 1000: loss 1.786149
    iteration 300 / 1000: loss 1.601071
    iteration 400 / 1000: loss 1.604213
    iteration 500 / 1000: loss 1.716027
    iteration 600 / 1000: loss 1.564976
    iteration 700 / 1000: loss 1.755343
    iteration 800 / 1000: loss 1.477814
    iteration 900 / 1000: loss 1.646744
    Processing H: 250 ::: Learning Rate: 0.001 ::: Reg Strength: 0.5
    iteration 0 / 1000: loss 2.306403
    iteration 100 / 1000: loss 1.867921
    iteration 200 / 1000: loss 1.792730
    iteration 300 / 1000: loss 1.611522
    iteration 400 / 1000: loss 1.616517
    iteration 500 / 1000: loss 1.731545
    iteration 600 / 1000: loss 1.586761
    iteration 700 / 1000: loss 1.775152
    iteration 800 / 1000: loss 1.497442
    iteration 900 / 1000: loss 1.665479
    Processing H: 250 ::: Learning Rate: 0.0001 ::: Reg Strength: 0.3
    iteration 0 / 1000: loss 2.304864
    iteration 100 / 1000: loss 2.302451
    iteration 200 / 1000: loss 2.280927
    iteration 300 / 1000: loss 2.182983
    iteration 400 / 1000: loss 2.083850
    iteration 500 / 1000: loss 2.103006
    iteration 600 / 1000: loss 2.040473
    iteration 700 / 1000: loss 2.043466
    iteration 800 / 1000: loss 1.914979
    iteration 900 / 1000: loss 2.000358
    Processing H: 250 ::: Learning Rate: 0.0001 ::: Reg Strength: 0.4
    iteration 0 / 1000: loss 2.305633
    iteration 100 / 1000: loss 2.303212
    iteration 200 / 1000: loss 2.281848
    iteration 300 / 1000: loss 2.184428
    iteration 400 / 1000: loss 2.085851
    iteration 500 / 1000: loss 2.105009
    iteration 600 / 1000: loss 2.042838
    iteration 700 / 1000: loss 2.045863
    iteration 800 / 1000: loss 1.918363
    iteration 900 / 1000: loss 2.003571
    Processing H: 250 ::: Learning Rate: 0.0001 ::: Reg Strength: 0.5
    iteration 0 / 1000: loss 2.306403
    iteration 100 / 1000: loss 2.303968
    iteration 200 / 1000: loss 2.282756
    iteration 300 / 1000: loss 2.185855
    iteration 400 / 1000: loss 2.087840
    iteration 500 / 1000: loss 2.106972
    iteration 600 / 1000: loss 2.045161
    iteration 700 / 1000: loss 2.048226
    iteration 800 / 1000: loss 1.921714
    iteration 900 / 1000: loss 2.006704
    Processing H: 250 ::: Learning Rate: 1e-05 ::: Reg Strength: 0.3
    iteration 0 / 1000: loss 2.304864
    iteration 100 / 1000: loss 2.304781
    iteration 200 / 1000: loss 2.304631
    iteration 300 / 1000: loss 2.304474
    iteration 400 / 1000: loss 2.304243
    iteration 500 / 1000: loss 2.304336
    iteration 600 / 1000: loss 2.303998
    iteration 700 / 1000: loss 2.303742
    iteration 800 / 1000: loss 2.303312
    iteration 900 / 1000: loss 2.303130
    Processing H: 250 ::: Learning Rate: 1e-05 ::: Reg Strength: 0.4
    iteration 0 / 1000: loss 2.305633
    iteration 100 / 1000: loss 2.305548
    iteration 200 / 1000: loss 2.305397
    iteration 300 / 1000: loss 2.305238
    iteration 400 / 1000: loss 2.305006
    iteration 500 / 1000: loss 2.305097
    iteration 600 / 1000: loss 2.304759
    iteration 700 / 1000: loss 2.304502
    iteration 800 / 1000: loss 2.304072
    iteration 900 / 1000: loss 2.303890
    Processing H: 250 ::: Learning Rate: 1e-05 ::: Reg Strength: 0.5
    iteration 0 / 1000: loss 2.306403
    iteration 100 / 1000: loss 2.306316
    iteration 200 / 1000: loss 2.306161
    iteration 300 / 1000: loss 2.306001
    iteration 400 / 1000: loss 2.305767
    iteration 500 / 1000: loss 2.305856
    iteration 600 / 1000: loss 2.305516
    iteration 700 / 1000: loss 2.305258
    iteration 800 / 1000: loss 2.304828
    iteration 900 / 1000: loss 2.304646
    


```python
# Print out results.
for h, lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(h, lr, reg)]
    print('h %s lr %e reg %e train accuracy: %f val accuracy: %f' % (
                h, lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
```

    h 130 lr 1.000000e-03 reg 5.000000e-01 train accuracy: 0.487510 val accuracy: 0.455000
    h 140 lr 1.000000e-03 reg 5.000000e-01 train accuracy: 0.490837 val accuracy: 0.481000
    h 150 lr 1.000000e-03 reg 5.000000e-01 train accuracy: 0.498694 val accuracy: 0.485000
    h 160 lr 1.000000e-03 reg 5.000000e-01 train accuracy: 0.494449 val accuracy: 0.470000
    best validation accuracy achieved during cross-validation: 0.485000
    


```python
# visualize the weights of the best network
show_net_weights(best_net)
```


![png](output_26_0.png)


# Run on the test set
When you are done experimenting, you should evaluate your final trained network on the test set; you should get above 48%.


```python
test_acc = (best_net.predict(X_test) == y_test).mean()
print('Test accuracy: ', test_acc)
```

    Test accuracy:  0.486
    

**Inline Question**

Now that you have trained a Neural Network classifier, you may find that your testing accuracy is much lower than the training accuracy. In what ways can we decrease this gap? Select all that apply.

1. Train on a larger dataset.
2. Add more hidden units.
3. Increase the regularization strength.
4. None of the above.

$\color{blue}{\textit Your Answer:}$

$\color{blue}{\textit Your Explanation:}$




```python

```
