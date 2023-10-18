# Analyzing Pruning Methods in Neural Networks and Their Influence on Accuracy
## Overview

Building large neural networks makes Neural Networks train successfully and perform better in various tasks. However, this makes them expensive, more time-consuming, and more challenging to distribute with more storage space. Therefore, Pruning was introduced to reduce the models' size by removing parameters from the existing network while maintaining accuracy. In other words, Pruning is a set of techniques that lower the computational demands of a neural network by removing weights, filters, neurons, or other structures.
## Project Background
[Jonathan Frankle](https://arxiv.org/pdf/1803.03635.pdf)  and Michael Carbin revolutionized the theoretical understanding of Neural Network pruning and identifying critical subnetworks within large neural networks in 2019 on the "Lottery Ticket Hypothesis" and "Sparse Networks from Scratch." Their hypothesis said that ''A randomly initialized, dense neural network contains a subnetwork
that is initialized such that—when trained in isolation—it can match the test accuracy of the original network after training for at most the same number of iterations''.

[Alex Renda ](https://arxiv.org/pdf/2003.02389.pdf) with his colleagues, discovered an alternative to fine-tuning (train the resultant pruned network): rewind the remaining weights to their values from earlier training and re-train the resulting network for the remainder of the original training process. They demonstrate the value of rewinding as a general pruning framework and compare rewinding and fine-tuning on CIFAR-10 and ImageNet networks. Their finding shows that a wide range of rewind points achieve higher accuracy than fine-tuning across all tested networks.


## Project Goal
In this work, I will present a structured pruning method (Filter) with a Group-Lasso regularization for CNNs where we remove whole network filters together with their connecting feature maps from CNNs. In addition, I will examine some unstructured pruning methods, like weight pruning, with different approaches and methods, like sparse training and fine-tuning. I will analyze if they improve model accuracy or are without significant accuracy loss. Moreover, if I have time, I will add some optimization techniques.
 ### Pruning Methods
### Datasets
1.[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
2.[ImageNet](https://paperswithcode.com/dataset/imagenet)

## Estimated Timeline
1. Design the Neural Network: 40h
2. Neural Network training and tuning: 20h
3. Implementation: 8h
4. Analysing and compering results: 4h
5. Report writing: 4h
