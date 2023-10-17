# Analyzing Pruning Methods in Neural Networks and Their Influence on Accuracy
## Overview

Building large neural networks makes Neural Networks train successfully and perform better in various tasks. However, this makes them expensive, more time-consuming, and more challenging to distribute with more storage space. Therefore, Pruning was introduced to reduce the models' size by removing parameters from the existing network while maintaining accuracy. In other words, Pruning is a set of techniques that lower the computational demands of a neural network by removing weights, filters, neurons, or other structures.
## Project Background
[Jonathan Frankle](https://arxiv.org/pdf/1803.03635.pdf)  and Michael Carbin revolutionized the theoretical understanding of Neural Network pruning and identifying critical subnetworks within large neural networks in 2019 on the "Lottery Ticket Hypothesis" and "Sparse Networks from Scratch." Their hypothesis said that ''A randomly initialized, dense neural network contains a subnetwork
that is initialized such that—when trained in isolation—it can match the test accuracy of the original network after training for at most the same number of iterations''.

[Alex Renda ](https://arxiv.org/pdf/2003.02389.pdf) with his colleagues, discovered an alternative to fine-tuning (train the resultant pruned network): rewind the remaining weights to their values from earlier training and re-train the resulting network for the remainder of the original training process. They demonstrate the value of rewinding as a general pruning framework and compare rewinding and fine-tuning on CIFAR-10 and ImageNet networks. Their finding shows that a wide range of rewind points achieve higher accuracy than fine-tuning across all tested networks.


## Project Goal
### Pruning Methods
## Estimated Timeline
