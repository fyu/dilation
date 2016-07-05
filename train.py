#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import os
from os.path import dirname, exists, join
import subprocess


__author__ = 'Fisher Yu'
__copyright__ = 'Copyright (c) 2016, Fisher Yu'
__email__ = 'i@yf.io'
__license__ = 'MIT'


def make_solver(options):
    solver = caffe_pb2.SolverParameter()

    solver.train_net = options.train_net
    if options.test_net is not None:
        solver.test_net.append(options.test_net)
        solver.test_iter.append(50)
    solver.test_interval = 100
    solver.base_lr = options.lr
    solver.lr_policy = "step"
    solver.gamma = 0.1
    solver.stepsize = 100000
    solver.display = 5
    solver.max_iter = 400000
    solver.momentum = 0.99
    solver.weight_decay = 0.0005
    solver.regularization_type = 'L2'
    solver.snapshot = 2000
    solver.solver_mode = solver.GPU
    solver.iter_size = 1
    solver.snapshot_format = solver.BINARYPROTO
    solver.type = 'SGD'
    solver.snapshot_prefix = options.snapshot_prefix

    return solver


def read_text_proto(file_path, parser_object):
    fp = open(file_path, "r")
    if not fp:
        raise IOError("Can not find {}".format(file_path))
    text_format.Merge(str(fp.read()), parser_object)
    fp.close()


def make_image_label_data(image_list_path, label_list_path, batch_size,
                          mirror, crop_size, mean_pixel):
    layer_param = caffe_pb2.LayerParameter()
    layer_param.name = 'data'
    layer_param.type = 'ImageLabelData'
    layer_param.top.extend(['data', 'label'])

    transform_param = layer_param.transform_param
    transform_param.mirror = mirror
    transform_param.crop_size = crop_size
    transform_param.mean_value.extend(mean_pixel)

    data_param = layer_param.image_label_data_param
    data_param.batch_size = batch_size
    data_param.shuffle = True
    data_param.rand_scale = False
    data_param.padding = caffe_pb2.ImageLabelDataParameter.REFLECT
    data_param.image_list_path = image_list_path
    data_param.label_list_path = label_list_path

    label_dim = (crop_size - 372) // 8
    label_slice = data_param.label_slice
    label_slice.offset.extend([186, 186])
    label_slice.stride.extend([8, 8])
    label_slice.dim.extend([label_dim, label_dim])

    return layer_param


def make_softmax_loss():
    layer_param = caffe_pb2.LayerParameter()
    layer_param.name = 'loss'
    layer_param.type = 'SoftmaxWithLoss'
    layer_param.bottom.extend(['final', 'label'])
    layer_param.top.append('loss')
    layer_param.loss_param.ignore_label = 255
    layer_param.loss_param.normalization = caffe_pb2.LossParameter.VALID

    return layer_param


def make_accuracy():
    layer_param = caffe_pb2.LayerParameter()
    layer_param.name = 'accuracy'
    layer_param.type = 'Accuracy'
    layer_param.bottom.extend(['final', 'label'])
    layer_param.top.append('accuracy')
    layer_param.accuracy_param.ignore_label = 255

    return layer_param


def make_nets(options):
    script_dir = dirname(__file__)
    template_dir = join(script_dir, 'templates')
    train_net = caffe_pb2.NetParameter()
    test_net = caffe_pb2.NetParameter()
    templ_path = join(template_dir, options.model + '_vgg.txt')
    read_text_proto(templ_path, train_net)
    test_net.CopyFrom(train_net)
    train_net.layer[0].CopyFrom(
        make_image_label_data(
            options.train_image, options.train_label,
            options.train_batch,
            True, options.crop_size, options.mean))
    train_net.layer.add().CopyFrom(make_softmax_loss())

    test_net.layer[0].CopyFrom(
        make_image_label_data(
            options.test_image, options.test_label, options.test_batch,
            False, options.crop_size, options.mean))
    test_net.layer.extend([make_softmax_loss(), make_accuracy()])

    return train_net, test_net


def process_options(options):
    assert (options.crop_size - 372) % 8 == 0, \
        "The crop size must be a multiple of 8 after removing the margin"
    assert len(options.mean) == 3

    work_dir = options.work_dir
    model = options.model
    if not exists(work_dir):
        print('Creating working directory', work_dir)
        os.makedirs(work_dir)
    options.train_net = join(work_dir, model + '_train_net.txt')
    options.test_net = join(work_dir, model + '_test_net.txt')
    options.solver_path = join(work_dir, model + '_solver.txt')
    snapshot_dir = join(work_dir, 'snapshots')
    if not exists(snapshot_dir):
        os.makedirs(snapshot_dir)
    options.snapshot_prefix = join(snapshot_dir, model)

    assert options.model == 'frontend', \
        'Only front end training is supported now'

    return options


def train(options):
    cmd = [options.caffe, 'train', '-solver', options.solver_path]
    if options.weights is not None:
        cmd.extend(['-weights', options.weights])
    subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', nargs='?',
                        choices=['frontend', 'context', 'joint'])
    parser.add_argument('--caffe', default='caffe',
                        help='Path to the caffe binary compiled from '
                             'https://github.com/fyu/caffe-dilation')
    parser.add_argument('--weights', default=None,
                        help='Path to the weights to initialize the model')
    parser.add_argument('--mean', nargs='*', default=[102.93, 111.36, 116.52],
                        help='Mean pixel value (BGR) for the dataset.\n'
                             'Default is the mean pixel of PASCAL dataset')
    parser.add_argument('--work_dir', default='training/',
                        help='working dir for training.')
    parser.add_argument('--train_image', default='',
                        help='Path to the training image list')
    parser.add_argument('--train_label', default='',
                        help='Path to the training label list')
    parser.add_argument('--test_image', default='',
                        help='Path to the testing image list')
    parser.add_argument('--test_label', default='',
                        help='Path to the testing label list')
    parser.add_argument('--train_batch', type=int, default=8,
                        help='Training batch size.')
    parser.add_argument('--test_batch', type=int, default=2,
                        help='Testing batch size. If it is 0, no test phase.')
    parser.add_argument('--crop_size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='Learning rate')

    options = parser.parse_args()
    options = process_options(options)

    train_net, test_net = make_nets(options)
    solver = make_solver(options)
    print('Writing', options.train_net)
    with open(options.train_net, 'w') as fp:
        fp.write(str(train_net))
    print('Writing', options.test_net)
    with open(options.test_net, 'w') as fp:
        fp.write(str(test_net))
    print('Writing', options.solver_path)
    with open(options.solver_path, 'w') as fp:
        fp.write(str(solver))
    train(options)


if __name__ == '__main__':
    main()
