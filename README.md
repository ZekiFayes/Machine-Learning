##  Machine Learning
This project is to build machine learning models -- computational models. Our goal is to go through machine learning, comprehend the concepts, and employ them to solve problems. That is to learn mathematics behind these algorithms. We will understand the basic concepts as well as implement the algorithms in Python/PyCharm.  

## Optimization Problem
Most of Machine Learning problems are Optimization problems, which means that we can find the solutions by minimizing or maximizing the cost function. If the machine learning problem is not an optimization problem. We can transform it into an optimization problem via transformation. Hence, we solve the machine learning problem by solving the optimization problem. Here comes optimization, epsecically convex optimization. Refer to Convex Optimization -- Stephen Boyd

## General Solution
Now that the machine learning problem can be an optimization problem. We can employ Back Propagation to update weights and biases. Some elements are involved in these process.

### Gradient Descent
Batch Gradient Descent,
Stochastic Gradient Descent,
Mini-batch Stochastic Gradient Descent.

### Update Strategy
Refer to An overview of gradient descent optimization algorithms -- Sebastian Ruder. 
Momentum,
Nesterov accelerated gradient,
Adagrad,
Adadelta,
RMSprop,
Adam.
  
## Common Problems
### Gradient Vanishing/Exploding
It usually ouccurs in the input layer. 

### Underfitting/Overfitting
It is very common.

## Common Solutions
Regularization. L1/L2 Norm, Dropout, Batch Normalization. 

## Basic Model
### Shallow Neural Network (SNN)
This is the very basic model for the complicated neural network.

### Convolutional Neural Network (CNN)
This model consists of Convolutional layer, pooling layer, fully-connected layer. This is especially for image processing.

### AutoEncoder (AE)
This model is used to make inputs sparse and denoise in the inputs. AE can be employed to pretrain deep neural network such that it is fast convergent.

### Recursive/Recurrent Neural Network (RNN)
This model is similar to other models, but it has a state or memory. In control engineering, we use modern control theory -- State Space Equation. Similarly, RNN is built based on this and solved by BPTT.

## Advanced Model
### Variational AutoEncoder (VAE)
This is a variant of AE which can be used to recover the inputs from noise. This is to construct an optimization problem by finding its lower bounds. Refer to Auto-Encoding Variational Bayes -- Diederik P. Kingma. 

### Generative Adversarial Network (GAN)
This is to train two networks at same time. One is Discriminator; the other is Generator. Refer to Generative Adversarial Nets -- Ian J. Goodfellow.

### Deep Convolutional GAN (DCGAN)
This is a variant of GAN. It uses Deep Convolutional Neural Network as a Generator. Refer to Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Network -- Alec Radford & Luke Metz.

## Architecture
#### LeNet5
#### AlexNet
#### VGG16/19
#### GoogleNet

Refer to Very deep convolutional networks for large-scale image recognition - Karen Simonya. Going Deeper with Convolutions - Christian Szegedy.

## Optimization Technique
### Initialization
This is to initialize the weights and biases. Refer to Understanding the difficulty of training deep feedforward neural networks -- Xavier Glorot.

### Batch Normalization (BN)
This is to normalize the data such that the gradients flow in and converge fast. The results show that BN makes the model stable. Refer to Batch Normalization: Accelerating Deep Neural Network Trainig by Reducing Internal Covariate Shift -- Sergey Ioffe.

### Knowledge Distillation (KD)
This is to compress a complicated model into a simle model. Refer to Distill the knowledge in a Neural Network -- Hinton.

### Pretraining (AE/RBM)
This is to pretrain neural network using AE or RBM. Refer to Reducing the Dimensionality of Data with Neural Network -- G. E. Hinton. A Fast Learning Algorithm for Deep Belief Nets -- Geoffrey E. Hinton.

### Dropout
This is to combine many trained models together such that it can prevent overfitting.  Refer to Dropout: A simple Way to Prevent Neural Network from Overfitting -- Nitish Srivastava.

## Probabilistic Graphical Model (PGM)

### Restricted Boltzmann Machine (RBM)
This is a undirected graphical model. It can be used in pretraining and recommendation systems. Refer to Boltzmann Machines -- Geoffrey E. Hinton. A practical Guide to training Restricted Boltzmann Machines.

### Directed Acyclic Graph (DAG)
This is a directed graphical model.

## Classification
#### Perceptron

#### Logistic Regression (LR)

#### Support vector Machine (SVM)

#### Decision Tree (DT)

#### Random Forest (RF)

#### K-Nearest Neighbor (KNN)

## Platform
### TensorFlow
The core elements are Tensors and Matrix Operations.
We refer to Numerical Methods for Computational Science and Engineering -- Prof. R. Hiptmair, SAM, ETH Zurich
