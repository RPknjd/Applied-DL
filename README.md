# Analysing Pruning Methods in Neural Networks and Their Influence on Accuracy
## Overview

Building large neural networks helps them train successfully and perform better in various tasks. However, this makes them expensive, more time-consuming, and more challenging to distribute with more storage space. Therefore, Pruning was introduced to reduce the models' size by removing parameters from the existing network while maintaining accuracy. In other words, Pruning is a set of techniques that lower the computational demands of a neural network by removing weights, filters, neurons, or other structures.
There are various methods for pruning a neural network, and the approach depends on what to prune when we would like to prune, and how to cope with this challenge of pruning parts without harming the network.

## Project Background

[Jonathan Frankle](https://arxiv.org/pdf/1803.03635.pdf) and Michael Carbin revolutionized the theoretical understanding of Neural Network pruning and identifying critical subnetworks within large neural networks in 2019 on the "Lottery Ticket Hypothesis" and "Sparse Networks from Scratch." Their hypothesis said that "A randomly initialized, dense neural network contains a subnetwork
that is initialized such that—when trained in isolation—it can match the test accuracy of the original network after training for at most the same number of iterations".

[Alex Renda ](https://arxiv.org/pdf/2003.02389.pdf), with his colleagues, discovered an alternative to fine-tuning (training the resultant pruned network): rewind the remaining weights to their values from earlier training and re-train the resulting network for the remainder of the original training process. They demonstrate the value of rewinding as a general pruning framework and compare rewinding and fine-tuning on CIFAR-10 and ImageNet networks. Their finding shows that many rewind points achieve higher accuracy than fine-tuning across all tested networks.

[Wei Wen ](https://arxiv.org/pdf/1608.03665.pdf) and his colleagues proposed a Structured Sparsity Learning (SSL) method to learn a compressed structure  (filters, channels, filter shapes, and layer depth) of DNNs by group Lasso regularization during the training. They stated that Group Lasso is an efficient regularization to learn sparse structures.

## Pruning structures
Structured and unstructured pruning methods are two different techniques for reducing the size of neural networks by removing unimportant weights, filters, or neurons.

### Structured Pruning
Structured Pruning removes a structure (building block) of the target neural network, such as a neuron for a Fully Connected Layer or a Channel of filter for a Convolution Layer. Structured Pruning means that by removing a particular network structure, we get (weight) matrices with smaller parameters (reduced size of parameters). There are some pruning criteria for structured Pruning to decide whether a neuron or channel of CNN is important or unimportant. Group Lasso can effectively zero out all weights in some groups in DNNs.
some groups

### Unstructured Pruning
Unstructured Pruning (magnitude pruning) converts some weights with smaller magnitudes into zeros. It means we convert an original dense (lots of non-zero values) network into a sparse (lots of zeros) network. The size of the weight matrix of the sparse network is the same as the size of the parameter matrix of the original network. However, the Sparse network has more zeros in its parameter matrix.

## Goals
This work will present a structured pruning method (filter) with a Group-Lasso regularization for CNNs. We remove whole network filters and their connecting feature maps from CNNs in the classification task. In addition, I will examine some unstructured pruning methods, like weight pruning, with different approaches and methods. I will analyze if they improve model accuracy or are without significant accuracy loss. Moreover, if I have time, I will add some optimization techniques.

1. Implementation of unstructured pruning method 
      - Implementation of weight pruning.
      - Calculation metrics (Accuracy, Loss, Size).
      - Compare the pruned model with the original model.
2. Implementation of structured pruning method
      - Implementation of NN training/testing task without Pruning.
      - Implementation of NN training/testing task with filter Pruning
  
3. Since I am a beginner in deep learning, gaining a better understanding of its structure and concepts through this work, is my personal goal.        
       
### Dataset
1.[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

## Estimated Timeline
1. Unstructured Pruning (Understanding its concept, set metrics. ):    Planned: 15h    Actual: 15h
2. Structured Pruning (Understanding its concept, set metrics. ):      Planned: 25h    Actual: 30h
3. Analysing results:       planned: 5h     Actual: 3h
4. Presentation:            planned: 10h     Actual: 
5. Application:             planned: 5h     Actual: 
6. Report:                  planned: 5h     Actual:

## Findings
1. A model with filter pruning and group lasso regularization is faster during training and testing. This is a positive outcome, as reduced computation time can be a significant advantage. The training and testing time also decreased.
2. The total sparsity across all layers is around 74.103%. Each layer has a different level of sparsity, with the convolutional layers being less sparse than the fully connected layers (fc1, fc2).

   


