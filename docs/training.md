# Training Dilation Network

The training code for context module and joint training is still in development. Please note that the code for training is still in development, so the usage may change in the next commit.


## Preparation

Besides the Python dependency listed in [README](https://github.com/fyu/dilation#python), you have to download and build my fork of Caffe from [fyu/caffe-dilation](https://github.com/fyu/caffe-dilation). It has new data layers to read and transform the raw images.


## Front End

Assume `${DILATION}` is the directory of [fyu/dilation](https://github.com/fyu/dilation) and `${CAFFE_DILATION_BUILD}` is the build directory for [fyu/caffe-dilation](https://github.com/fyu/caffe-dilation).

The code used for training is `${DILATION}/train.py`. It takes some parameters for the data layers and tries to fill in the other automatically.

Before training the front end, please download the weights of VGG network trained on ImageNet

```bash
sh ${DILATION}/pretrained/download_vgg_conv.sh
```

train.py takes different options to make it flexible to train the model with different parameters. Four of the parameters are used to read the input data:

```
  --train_image TRAIN_IMAGE
                        Path to the training image list
  --train_label TRAIN_LABEL
                        Path to the training label list
  --test_image TEST_IMAGE
                        Path to the testing image list
  --test_label TEST_LABEL
                        Path to the testing label list
```

Please note that training/testing image/label lists are text files, in which each line specify a file path to the input or label image for training. The image and label lists for training or testing should have the same number of lines. The labels and images are corresponded by line number. The testing data is for the test phase of Caffe. Normally, it refers to the validation set in a dataset.

Sometimes, it is critical to set the right learning rate and momentum to get the best training results. `train.py` tries to set some reasonable values by default, but the optimal setting depends on the image and dataset set. Please refer to [our paper](https://arxiv.org/abs/1511.07122) for settings on different datasets. To change the other solver parameters, please check the function `make_solver` in `train.py`.

Below is an example to train front end on PASCAL VOC dataset:

```bash
python ${DILATION}/train.py frontend \
--work_dir training \
--train_image <path to training image list> \
--train_label <path to traiing label list> \
--test_image <path to testing image list> \
--test_label <path to testing label list> \
--train_batch 14 \
--test_batch 2 \
--caffe ${CAFFE_DILATION_BUILD}/tools/caffe \
--weights ${DILATION}/pretrained/vgg_conv.caffemodel \
--crop_size 500 \
--classes 21
--lr 0.0001
--momentum 0.9
```

After the training procedure finishes,  `test.py` can generate the prediction results from a list of images based on the trained caffe model. As with `train.py`, it will save network definition for deploy in `work_dir`. If continuing to train context module, you can add `--bin` to the command line to extract the responses of the last feature layer, which will serve as input to the context module. After processing all the images, a list of generated features will be written to the `feat` folder in `work_dir`, which can serve as input for training context module.

```bash
python ${DILATION}/test.py frontend \
--work_dir training \
--image_list <image_list> \
--weights <caffe_model_path> \
--classes 21 \
--bin
```

`test.py` can also be used to generating prediction results for context module and joint training.

##Context Module

Similar to front end, `train.py` can be used for training context module based on the ouput of `test.py`.  Among the parameters, `layers` specifies the number of layers in context module depending on the input image size. 8 is good for PASCAL VOC dataset. `label_shape` is the height and weight of the stored features for each image.

Here is an example to train the context module with `train.py`

```bash
python ${DILATION}/train.py context \
--train_image <path to training feature bin list> \
--train_label <path to traiing label list> \
--test_image <path to testing feature bin list> \
--test_label <path to testing label list> \
--train_batch 60 \
--test_batch 10 \
--caffe ${CAFFE_DILATION_BUILD}/tools/caffe \
--classes 21 \
--layers 8 \
--label_shape 66 66
--lr 0.001
--momentum 0.9
```

##Joint Training

After training the context module, it can sometimes improve the results further to train the front end and context module jointly. If the dataset only has hundreds of images, it is possible to skip training context module and do joint training directly with identity initialization for context module.

Example to train jointly

```bash
python ${DILATION}/train.py joint \
--train_image <path to training image list> \
--train_label <path to traiing label list> \
--test_image <path to testing image list> \
--test_label <path to testing label list> \
--train_batch 14 \
--test_batch 2 \
--caffe ${CAFFE_DILATION_BUILD}/tools/caffe \
--weights <trained frontend model>,<trained context model> \
--classes 21 \
--layers 8
--lr 0.00001
--momentum 0.9
```