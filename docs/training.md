# Training Dilation Network

The training code for context module and joint training is still in development. Please note that the code for training is still in development, so the usage may change in the next commit.


## Preparation

Besides the Python dependency listed in [README](https://github.com/fyu/dilation#python), you have to download and build my fork of Caffe from [fyu/caffe-dilation](https://github.com/fyu/caffe-dilation). It has new data layers to read and transform the raw images.


## Front End

Assume ${DILATION} is the directory of [fyu/dilation](https://github.com/fyu/dilation) and ${CAFFE_DILATION_BUILD} is the build directory for [fyu/caffe-dilation](https://github.com/fyu/caffe-dilation).

The code used for training is ${DILATION}/train.py. It takes some parameters for the data layers and tries to fill in the other automatically.

Before training the front end, please download the weights of VGG network trained on ImageNet

```bash
sh ${DILATION}/pretrained/download_vgg_conv.sh
```

The script document:

```bash
usage: train.py [-h] [--caffe CAFFE] [--weights WEIGHTS]
                [--mean [MEAN [MEAN ...]]] [--work_dir WORK_DIR] --train_image
                TRAIN_IMAGE --train_label TRAIN_LABEL
                [--test_image TEST_IMAGE] [--test_label TEST_LABEL]
                [--train_batch TRAIN_BATCH] [--test_batch TEST_BATCH]
                [--crop_size CROP_SIZE] [--lr LR]
                [{frontend,context,joint}]

positional arguments:
  {frontend,context,joint}

optional arguments:
  -h, --help            show this help message and exit
  --caffe CAFFE         Path to the caffe binary compiled from
                        https://github.com/fyu/caffe-dilation.
  --weights WEIGHTS     Path to the weights to initialize the model.
  --mean [MEAN [MEAN ...]]
                        Mean pixel value (BGR) for the dataset. Default is the
                        mean pixel of PASCAL dataset.
  --work_dir WORK_DIR   Working dir for training. All the generated network
                        and solver configurations will be written to this
                        directory, in addition to training snapshots.
  --train_image TRAIN_IMAGE
                        Path to the training image list
  --train_label TRAIN_LABEL
                        Path to the training label list
  --test_image TEST_IMAGE
                        Path to the testing image list
  --test_label TEST_LABEL
                        Path to the testing label list
  --train_batch TRAIN_BATCH
                        Training batch size.
  --test_batch TEST_BATCH
                        Testing batch size. If it is 0, no test phase.
  --crop_size CROP_SIZE
  --lr LR               Solver earning rate
```

Please note that training/testing image/label lists are text files, in which each line specify a file path to the input or label image for training. The image and label lists for training or testing should have the same number of lines. The labels and images are corresponded by line number. The testing data is for the test phase of Caffe. Normally, it refers to the validation set in a dataset.

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
--crop_size 500
```
