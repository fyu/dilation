# Multi-Scale Context Aggregation by Dilated Convolutions

## Introduction

Properties of dilated convolution are discussed in our [ICLR 2016 conference paper](http://arxiv.org/abs/1511.07122). This repository contains the network definitions and the trained models. You can use this code together with vanilla Caffe to segment images using the pre-trained models. If you want to train the models yourself, please check out the [ducoment for training](https://github.com/fyu/dilation/blob/master/docs/training.md).

### Citing

If you find the code or the models useful, please cite this paper:
```
@inproceedings{YuKoltun2016,
	author    = {Fisher Yu and Vladlen Koltun},
	title     = {Multi-Scale Context Aggregation by Dilated Convolutions},
	booktitle = {ICLR},
	year      = {2016},
}
```
### License

The code and models are released under the MIT License (refer to the LICENSE file for details).


## Installation
### Caffe

Install [Caffe](https://github.com/BVLC/caffe) and its Python interface. Make sure that the Caffe version is newer than commit [08c5df](https://github.com/BVLC/caffe/commit/08c5dfd53e6fd98148d6ce21e590407e38055984).

### Python

The companion Python script is used to demonstrate the network definition and trained weights.

The required Python packages are numba numpy opencv. Python release from Anaconda is recommended.

In the case of using Anaconda
```
conda install numba numpy opencv
```

## Running Demo

predict.py is the main script to test the pre-trained models on images. The basic usage is

    python predict.py <dataset name> <image path>

Given the dataset name, the script will find the pre-trained model and network definition. We currently support models trained from four datasets: pascal_voc, camvid, kitti, cityscapes. The steps of using the code is listed below:

* Clone the code from Github

    ```
    git clone git@github.com:fyu/dilation.git
    cd dilation
    ```
* Download pre-trained network

    ```
    sh pretrained/download_pascal_voc.sh
    ```
* Run pascal voc model on GPU 0

    ```
    python predict.py pascal_voc images/dog.jpg --gpu 0
    ```
    
## Training

You are more than welcome to train our model on a new dataset. To do that, please refer to the [document for training](docs/training.md).

## Implementation of Dilated Convolution

Besides Caffe support, dilated convolution is also implemented in other deep learning packages. For example,
* Torch: [SpatialDilatedConvolution](https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialDilatedConvolution)
* Lasagne: [DilatedConv2DLayer](http://lasagne.readthedocs.io/en/latest/modules/layers/conv.html?highlight=dilated#lasagne.layers.DilatedConv2DLayer)
