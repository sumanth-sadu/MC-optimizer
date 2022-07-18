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

# How to use
To run on cifar-10, cifar-100 datasets </br> 
run **cifar_notebook.ipynb** changing the file paths accordingly </br>
To run on Tiny-imagenet dataset </br>
run **tiny_imagenet_notebook.ipynb** changing the file paths accordingly </br>

# Citation
Sumanth Sadu, Shiv Ram Dubey, and S. R. Sreeja, "Moment Centralization based Gradient Descent Optimizers for Convolutional Neural Networks", International Conference on Computer Vision and Machine Intelligence, 2022.

# Datasets
Cifar-10, Cifar-100 are downloaded from torchvision. </br>
Tiny-imagenet dataset can be downloaded from http://cs231n.stanford.edu/tiny-imagenet-200.zip </br>
For Tiny-imagenet dataset formating and resizing of data is required which can be done using val_format.py and resize.py files. 

# References
Code used from [pytorch-cifar-models](https://github.com/junyuseu/pytorch-cifar-models)
