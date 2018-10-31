# Smooth Neighbors on Teacher Graphs for Semi-supervised Learning (CVPR 2018 spotlight)
This is an implementation of SNTG reproducing the results in **<a href="https://arxiv.org/pdf/1711.00258">Smooth Neighbors on Teacher Graphs for Semi-supervised Learning.
</a>**

By [Yucen Luo](http://bigml.cs.tsinghua.edu.cn/~yucen/), [Jun Zhu](http://ml.cs.tsinghua.edu.cn/~jun/index.shtml), Mengxi Li, Yong Ren, Bo Zhang.

Most code is adapted from [Temporal Ensembling](https://github.com/smlaine2/tempens)

## Quick start
Reproducing the results on MNIST, SVHN and CIFAR10.

Modify the hyperparameters in config.py and then ``` python train_emb.py ```

## Requirements 
- Theano 0.9.0.dev4
- Lasagne 0.2.dev1
- CUDA toolkit 8.0, CUDNN 5105


