# Moment-Centralization-based-Gradient-Descent-Optimizers-for-Convolutional-Neural-Networks
This repo contains the code used in Moment Centralization based Gradient Descent Optimizers for Convolutional Neural Networks paper. An optimizer for convolutional neural networks. 

Convolutional neural networks (CNNs) have shown very appealing
performance for many computer vision applications. The training of CNNs is
generally performed using stochastic gradient descent (SGD) based optimization techniques. The adaptive momentum-based SGD optimizers are the recent
trends. However, the existing optimizers are not able to maintain a zero mean
in the first-order moment and struggle with optimization. In this paper, we propose a moment centralization-based SGD optimizer for CNNs. Specifically, we
impose the zero mean constraints on the first-order moment explicitly. The proposed moment centralization is generic in nature and can be integrated with any
of the existing adaptive momentum-based optimizers. The proposed idea is tested
with three state-of-the-art optimization techniques, including Adam, Radam, and
Adabelief on benchmark CIFAR10, CIFAR100, and TinyImageNet datasets for
image classification. The performance of the existing optimizers is generally improved when integrated with the proposed moment centralization. Further, The
results of the proposed moment centralization are also better than the existing
gradient centralization. The analytical analysis using the toy example shows that
the proposed method leads to a shorter and smoother optimization trajectory.
