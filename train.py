#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import caffe
from caffe.proto import caffe_pb2
import os
from os.path import dirname, exists, join
import subprocess

import network


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


def make_net(options, is_training):
    net = caffe.NetSpec()
    net.data, net.label = network.make_image_label_data(
        options.train_image, options.train_label, options.train_batch,
        is_training, options.crop_size, options.mean)
    last = network.build_frontend_vgg(
        net, net.data, options.classes)[0]
    if options.up:
        net.upsample = network.make_upsample(last, options.classes)
        last = net.upsample
    net.loss = network.make_softmax_loss(last, net.label)
    if not is_training:
        net.accuracy = network.make_accuracy(last, net.label)
    net = net.to_proto()
    return net


def make_nets(options):
    train_net = make_net(options, True)
    if options.test_net is None:
        test_net = None
    else:
        test_net = make_net(options, False)
    return train_net, test_net


def process_options(options):
    assert (options.crop_size - 372) % 8 == 0, \
        "The crop size must be a multiple of 8 after removing the margin"
    assert len(options.mean) == 3

    assert options.model == 'frontend', \
        'Only front end training is supported now'

    assert exists(options.train_image), options.train_image + 'does not exist'
    assert exists(options.train_label), options.train_label + 'does not exist'
    assert exists(options.test_image), options.test_image + 'does not exist'
    assert exists(options.test_label), options.test_label + 'does not exist'

    if options.model == 'frontend':
        options.model += '_vgg'

    work_dir = options.work_dir
    model = options.model
    if not exists(work_dir):
        print('Creating working directory', work_dir)
        os.makedirs(work_dir)
    options.train_net = join(work_dir, model + '_train_net.txt')
    if options.test_batch > 0:
        options.test_net = join(work_dir, model + '_test_net.txt')
    else:
        options.test_net = None
    options.solver_path = join(work_dir, model + '_solver.txt')
    snapshot_dir = join(work_dir, 'snapshots')
    if not exists(snapshot_dir):
        os.makedirs(snapshot_dir)
    options.snapshot_prefix = join(snapshot_dir, model)

    return options


def train(options):
    cmd = [options.caffe, 'train', '-solver', options.solver_path,
           '-gpu', options.gpu]
    if options.weights is not None:
        cmd.extend(['-weights', options.weights])
    subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', nargs='?',
                        choices=['frontend', 'context', 'joint'])
    parser.add_argument('--caffe', default='caffe',
                        help='Path to the caffe binary compiled from '
                             'https://github.com/fyu/caffe-dilation.')
    parser.add_argument('--weights', default=None,
                        help='Path to the weights to initialize the model.')
    parser.add_argument('--mean', nargs='*', default=[102.93, 111.36, 116.52],
                        help='Mean pixel value (BGR) for the dataset.\n'
                             'Default is the mean pixel of PASCAL dataset.')
    parser.add_argument('--work_dir', default='training/',
                        help='Working dir for training.\nAll the generated '
                             'network and solver configurations will be written '
                             'to this directory, in addition to training '
                             'snapshots.')
    parser.add_argument('--train_image', default='', required=True,
                        help='Path to the training image list')
    parser.add_argument('--train_label', default='', required=True,
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
                        help='Solver earning rate')
    parser.add_argument('--classes', type=int, required=True,
                        help='Number of categories in the data')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU index for training')
    parser.add_argument('--up', action='store_true',
                        help='If true, upsampling the final feature map '
                             'before calculating the loss or accuracy')

    options = process_options(parser.parse_args())

    train_net, test_net = make_nets(options)
    solver = make_solver(options)
    print('Writing', options.train_net)
    with open(options.train_net, 'w') as fp:
        fp.write(str(train_net))
    if test_net is not None:
        print('Writing', options.test_net)
        with open(options.test_net, 'w') as fp:
            fp.write(str(test_net))
    print('Writing', options.solver_path)
    with open(options.solver_path, 'w') as fp:
        fp.write(str(solver))
    train(options)


if __name__ == '__main__':
    main()
