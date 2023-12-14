# Analysing Pruning Methods in Neural Networks and Their Influence on Accuracy
## Overview

Building large neural networks helps them train successfully and perform better in various tasks. However, this makes them expensive, more time-consuming, and more challenging to distribute with more storage space. Therefore, Pruning was introduced to reduce the models' size by removing parameters from the existing network while maintaining accuracy. In other words, Pruning is a set of techniques that lower the computational demands of a neural network by removing weights, filters, neurons, or other structures.
There are various methods for pruning a neural network, and the approach depends on what to prune, when we would like to prune, and how to cope with this challenge of pruning parts without harming the network.

## Project Background

[Jonathan Frankle](https://arxiv.org/pdf/1803.03635.pdf) and Michael Carbin revolutionized the theoretical understanding of Neural Network pruning and identifying critical subnetworks within large neural networks in 2019 on the "Lottery Ticket Hypothesis" and "Sparse Networks from Scratch." Their hypothesis said that "A randomly initialized, dense neural network contains a subnetwork
that is initialized such that—when trained in isolation—it can match the test accuracy of the original network after training for at most the same number of iterations".

[Alex Renda ](https://arxiv.org/pdf/2003.02389.pdf) with his colleagues, discovered an alternative to fine-tuning (train the resultant pruned network): rewind the remaining weights to their values from earlier training and re-train the resulting network for the remainder of the original training process. They demonstrate the value of rewinding as a general pruning framework and compare rewinding and fine-tuning on CIFAR-10 and ImageNet networks. Their finding shows that a wide range of rewind points achieve higher accuracy than fine-tuning across all tested networks.

[Wei Wen ](https://arxiv.org/pdf/1608.03665.pdf) and his colleagues proposed a Structured Sparsity Learning (SSL) method to learn a compressed structure  (filters, channels, filter shapes, and layer depth) of DNNs by group Lasso regularization during the training. They stated that Group Lasso is an efficient regularization to learn sparse structures.

## Pruning structures
Structured and unstructured pruning methods are two different techniques for reducing the size of neural networks by removing unimportant weights, filters, or neurons.

### Structured pruning
Structured pruning removes a structure (building block) of the target neural network, such as, Neuron for a Fully Connected Layer or Channel of filter for a Convolution Layer and etc. Structured pruning means that by removing a particular structure of a network, we get (weight) matrices with smaller parameters (reduced size of parameters). there are some pruning criteria for structured pruning to decide whether a neuron or channel of CNN is important or unimportant. Group Lasso can effectively zero out all weights in some groups in DNNs.
some groups

### Unstructured pruning
Unstructured pruning (magnitude pruning) converts some of the weights with smaller magnitude into zeros.  It means that we converts an original dense (lots of non-zero values) network into a sparse (lots of zeros) network. The size of the weight matrix of the sparse network is the same as the size of parameter matrix of the original network. but Sparse network has more zeros in their parameter matrix.

## Project Goal
In this work, I will present a structured pruning method (Filter) with a Group-Lasso regularization for CNNs where we remove whole network filters together with their connecting feature maps from CNNs in classification task. In addition, I will examine some unstructured pruning methods, like weight pruning, with different approaches and methods. I will analyse if they improve model accuracy or are without significant accuracy loss. Moreover, if I have time, I will add some optimization techniques.

 
### Dataset
1.[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)


## Estimated Timeline
1. Design the Neural Network: 40h
2. Neural Network training and tuning: 25h
3. Implementation: 8h
4. Analysing and compering results: 5h
5. Report writing: 5h
6. preparing presentaion: 5h

## References


## Useful links
1. [Pruning Neural Networks](https://pohsoonchang.medium.com/neural-network-pruning-update-cda56343e5a2)
