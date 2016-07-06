#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
from caffe import layers as L
from caffe import params as P


__author__ = 'Fisher Yu'
__copyright__ = 'Copyright (c) 2016, Fisher Yu'
__email__ = 'i@yf.io'
__license__ = 'MIT'


def make_image_label_data(image_list_path, label_list_path, batch_size,
                          mirror, crop_size, mean_pixel,
                          label_stride=8, margin=186):
    label_dim = (crop_size - margin * 2) // 8
    data, label = L.ImageLabelData(
        transform_param=dict(mirror=mirror, mean_value=mean_pixel,
                             crop_size=crop_size),
        image_label_data_param=dict(
            image_list_path=image_list_path, label_list_path=label_list_path,
            shuffle=True, batch_size=batch_size,
            padding=P.ImageLabelData.REFLECT,
            label_slice=dict(dim=[label_dim, label_dim],
                             stride=[label_stride, label_stride],
                             offset=[margin, margin])),
        ntop=2)
    return data, label


def make_bin_label_data(bin_list_path, label_list_path, batch_size,
                        label_shape, label_stride):
    data, label = L.BinLabelData(
        bin_label_data_param=dict(
            bin_list_path=bin_list_path, label_list_path=label_list_path,
            shuffle=True, batch_size=batch_size,
            label_slice=dict(stride=[label_stride, label_stride],
                             dim=label_shape)),
        ntop=2)
    return data, label


def make_input_data(input_size, channels=3):
    return L.Input(input_param=dict(shape=dict(
        dim=[1, channels, input_size[0], input_size[1]])))


def make_softmax_loss(bottom, label):
    return L.SoftmaxWithLoss(bottom, label,
                             loss_param=dict(ignore_label=255,
                                             normalization=P.Loss.VALID))


def make_accuracy(bottom, label):
    return L.Accuracy(bottom, label, accuracy_param=dict(ignore_label=255))


def make_prob(bottom):
    return L.Softmax(bottom)


def make_upsample(bottom, num_classes):
    return L.Deconvolution(
        bottom,
        param=[dict(lr_mult=0, decay_mult=0)],
        convolution_param=dict(
            bias_term=False, num_output=num_classes, kernel_size=16, stride=8,
            group=num_classes, pad=4, weight_filler=dict(type="bilinear")))


def build_frontend_vgg(net, bottom, num_classes):
    prev_layer = bottom
    num_convolutions = [2, 2, 3, 3, 3]
    dilations = [0, 0, 0, 0, 2, 4]
    for l in range(5):
        num_outputs = min(64 * 2 ** l, 512)
        for i in range(0, num_convolutions[l]):
            conv_name = 'conv{0}_{1}'.format(l+1, i+1)
            relu_name = 'relu{0}_{1}'.format(l+1, i+1)
            if dilations[l] == 0:
                setattr(net, conv_name,
                        L.Convolution(
                            prev_layer,
                            param=[dict(lr_mult=1, decay_mult=1),
                                   dict(lr_mult=2, decay_mult=0)],
                            convolution_param=dict(num_output=num_outputs,
                                                   kernel_size=3)))
            else:
                setattr(net, conv_name,
                        L.Convolution(
                            prev_layer,
                            param=[dict(lr_mult=1, decay_mult=1),
                                   dict(lr_mult=2, decay_mult=0)],
                            convolution_param=dict(num_output=num_outputs,
                                                   kernel_size=3,
                                                   dilation=dilations[l])))
            setattr(net, relu_name,
                    L.ReLU(getattr(net, conv_name), in_place=True))
            prev_layer = getattr(net, relu_name)
        if dilations[l+1] == 0:
            pool_name = 'pool{0}'.format(l+1)
            setattr(net, pool_name, L.Pooling(
                prev_layer, pool=P.Pooling.MAX, kernel_size=2, stride=2))
            prev_layer = getattr(net, pool_name)

    net.fc6 = L.Convolution(
        prev_layer,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        convolution_param=dict(num_output=4096, kernel_size=7,
                               dilation=dilations[5]))
    net.relu6 = L.ReLU(net.fc6, in_place=True)
    net.drop6 = L.Dropout(net.relu6, in_place=True, dropout_ratio=0.5)
    net.fc7 = L.Convolution(
        net.drop6,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        convolution_param=dict(num_output=4096, kernel_size=1))
    net.relu7 = L.ReLU(net.fc7, in_place=True)
    net.drop7 = L.Dropout(net.relu7, in_place=True, dropout_ratio=0.5)
    net.final = L.Convolution(
        net.drop7,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        convolution_param=dict(
            num_output=num_classes, kernel_size=1,
            weight_filler=dict(type='gaussian', std=0.001),
            bias_filler=dict(type='constant', value=0)))
    return net.final, 'final'


def build_context(net, bottom, num_classes, layers=8):
    prev_layer = bottom
    multiplier = 1
    for i in range(1, 3):
        conv_name = 'ctx_conv1_{}'.format(i)
        relu_name = 'ctx_relu1_{}'.format(i)
        setattr(net, conv_name,
                L.Convolution(
                    *([] if prev_layer is None else [prev_layer]),
                    param=[dict(lr_mult=1, decay_mult=1),
                           dict(lr_mult=2, decay_mult=0)],
                    convolution_param=dict(
                        num_output=num_classes * multiplier, kernel_size=3,
                        pad=1,
                        weight_filler=dict(type='identity',
                                           num_groups=num_classes, std=0.01),
                        bias_filler=dict(type='constant', value=0))))
        setattr(net, relu_name,
                L.ReLU(getattr(net, conv_name), in_place=True))
        prev_layer = getattr(net, relu_name)

    for i in range(2, layers - 2):
        dilation = 2 ** (i - 1)
        multiplier = 1
        conv_name = 'ctx_conv{}_1'.format(i)
        relu_name = 'ctx_relu{}_1'.format(i)
        setattr(net, conv_name,
                L.Convolution(
                    prev_layer,
                    param=[dict(lr_mult=1, decay_mult=1),
                           dict(lr_mult=2, decay_mult=0)],
                    convolution_param=dict(
                        num_output=num_classes * multiplier, kernel_size=3,
                        dilation=dilation, pad=dilation,
                        weight_filler=dict(type='identity',
                                           num_groups=num_classes,
                                           std=0.01 / multiplier),
                        bias_filler=dict(type='constant', value=0))))
        setattr(net, relu_name,
                L.ReLU(getattr(net, conv_name), in_place=True))
        prev_layer = getattr(net, relu_name)

    net.ctx_fc1 = L.Convolution(
        prev_layer,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        convolution_param=dict(
            num_output=num_classes * multiplier, kernel_size=3, pad=1,
            weight_filler=dict(type='identity',
                               num_groups=num_classes,
                               std=0.01 / multiplier),
            bias_filler=dict(type='constant', value=0)))
    net.ctx_fc1_relu = L.ReLU(net.ctx_fc1, in_place=True)
    net.ctx_final = L.Convolution(
        net.ctx_fc1_relu,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        convolution_param=dict(
            num_output=num_classes, kernel_size=1,
            weight_filler=dict(type='identity',
                               num_groups=num_classes,
                               std=0.01 / multiplier),
            bias_filler=dict(type='constant', value=0)))
    return net.ctx_final, 'ctx_final'
