# Multi-Scale Context Aggregation by Dilated Convolutions

## Introduction

Properties of dilated convolution are discussed in the [arXiv report](http://arxiv.org/abs/1511.07122) accepted as ICLR 2016 conference paper. It can be used for semantic image segmentation and learning context information. This repo releases the network definition discussed in the report and the trained models.

### Citing

If you find the code or model useful in your research, please consider citing:
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

Install [Caffe](https://github.com/BVLC/caffe) and its Python interface. Make sure Caffe version is newer than commit [08c5df](https://github.com/BVLC/caffe/commit/08c5dfd53e6fd98148d6ce21e590407e38055984).

### Python

The companion Python script is used to demo the network definition and trained weights.

The required Python packages are numba numpy opencv. Python release from Anaconda is recommended. 

In the case of using Anaconda
```
conda install numba numpy opencv
```

## Running Demo

predict.py is the main script to test the pretrained model on images. The basic usage is
    
    python predict.py <dataset name> <image path>

Given the dataset name, the script will find the pretrained model and network definition. We currently support models trained from 4 datasets: pascal_voc, camvid, kitti, cityscapes. The steps of using the code is listed below:

* Clone the code from Github	
    
    ```
    git clone git@github.com:fyu/dilation.git
    cd dilation
    ```
* Download pretrained network
	
    ```
    sh pretrained/download_pascal_voc.sh
    ```
* Run pascal voc model on GPU 0
	
    ```
    python predict.py pascal_voc images/dog.jpg --gpu 0
    ```
