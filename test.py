#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import caffe
import cv2
import numpy as np
import os
from os.path import exists, join, split, splitext

import network
import util

__author__ = 'Fisher Yu'
__copyright__ = 'Copyright (c) 2016, Fisher Yu'
__email__ = 'i@yf.io'
__license__ = 'MIT'


def read_array(filename):
    with open(filename, 'rb') as fp:
        type_code = np.fromstring(fp.read(4), dtype=np.int32)
        shape_size = np.fromstring(fp.read(4), dtype=np.int32)
        shape = np.fromstring(fp.read(4 * shape_size), dtype=np.int32)
        if type_code == cv2.CV_32F:
            dtype = np.float32
        if type_code == cv2.CV_64F:
            dtype = np.float64
        return np.fromstring(fp.read(), dtype=dtype).reshape(shape)


def write_array(filename, array):
    with open(filename, 'wb') as fp:
        if array.dtype == np.float32:
            typecode = cv2.CV_32F
        elif array.dtype == np.float64:
            typecode = cv2.CV_64F
        else:
            raise ValueError("type is not supported")
        fp.write(np.array(typecode, dtype=np.int32).tostring())
        fp.write(np.array(len(array.shape), dtype=np.int32).tostring())
        fp.write(np.array(array.shape, dtype=np.int32).tostring())
        fp.write(array.tostring())


def make_frontend_vgg(options):
    deploy_net = caffe.NetSpec()
    deploy_net.data = network.make_input_data(options.input_size)
    last, final_name = network.build_frontend_vgg(
        deploy_net, deploy_net.data, options.classes)
    if options.up:
        deploy_net.upsample = network.make_upsample(last, options.classes)
        last = deploy_net.upsample
    deploy_net.prob = network.make_prob(last)
    deploy_net = deploy_net.to_proto()
    return deploy_net, final_name


def make_context(options):
    deploy_net = caffe.NetSpec()
    deploy_net.data = network.make_input_data(
        options.input_size, options.classes)
    last, final_name = network.build_context(
        deploy_net, deploy_net.data, options.classes, options.layers)
    if options.up:
        deploy_net.upsample = network.make_upsample(last, options.classes)
        last = deploy_net.upsample
    deploy_net.prob = network.make_prob(last)
    deploy_net = deploy_net.to_proto()
    return deploy_net, final_name


def make_joint(options):
    deploy_net = caffe.NetSpec()
    deploy_net.data = network.make_input_data(options.input_size)
    last = network.build_frontend_vgg(
        deploy_net, deploy_net.data, options.classes)[0]
    last, final_name = network.build_context(
        deploy_net, last, options.classes, options.layers)
    if options.up:
        deploy_net.upsample = network.make_upsample(last, options.classes)
        last = deploy_net.upsample
    deploy_net.prob = network.make_prob(last)
    deploy_net = deploy_net.to_proto()
    return deploy_net, final_name


def make_deploy(options):
    return globals()['make_' + options.model](options)


def test_image(options):
    options.feat_dir = join(options.feat_dir, options.feat_layer_name)
    if not exists(options.feat_dir):
        os.makedirs(options.feat_dir)

    label_margin = 186

    if options.up:
        zoom = 1
    else:
        zoom = 8

    if options.gpu >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(options.gpu)
        print('Using GPU ', options.gpu)
    else:
        caffe.set_mode_cpu()
        print('Using CPU')

    mean_pixel = np.array(options.mean, dtype=np.float32)
    net = caffe.Net(options.deploy_net, options.weights, caffe.TEST)

    image_paths = [line.strip() for line in open(options.image_list, 'r')]
    image_names = [split(p)[1] for p in image_paths]
    input_dims = list(net.blobs['data'].shape)

    assert input_dims[0] == 1
    batch_size, num_channels, input_height, input_width = input_dims
    print('Input size:', input_dims)
    caffe_in = np.zeros(input_dims, dtype=np.float32)

    output_height = input_height - 2 * label_margin
    output_width = input_width - 2 * label_margin

    result_list = []
    feat_list = []

    for i in range(len(image_names)):
        print('Predicting', image_names[i])
        image = cv2.imread(image_paths[i]).astype(np.float32) - mean_pixel
        image_size = image.shape
        print('Image size:', image_size)
        image = cv2.copyMakeBorder(image, label_margin, label_margin,
                                   label_margin, label_margin,
                                   cv2.BORDER_REFLECT_101)
        num_tiles_h = image_size[0] // output_height + \
                      (1 if image_size[0] % output_height else 0)
        num_tiles_w = image_size[1] // output_width + \
                      (1 if image_size[1] % output_width else 0)
        prediction = []
        feat = []
        for h in range(num_tiles_h):
            col_prediction = []
            col_feat = []
            for w in range(num_tiles_w):
                offset = [output_height * h,
                          output_width * w]
                tile = image[offset[0]:offset[0] + input_height,
                             offset[1]:offset[1] + input_width, :]
                margin = [0, input_height - tile.shape[0],
                          0, input_width - tile.shape[1]]
                tile = cv2.copyMakeBorder(tile, margin[0], margin[1],
                                          margin[2], margin[3],
                                          cv2.BORDER_REFLECT_101)
                caffe_in[0] = tile.transpose([2, 0, 1])
                blobs = []
                if options.bin:
                    blobs = [options.feat_layer_name]
                out = net.forward_all(blobs=blobs, **{net.inputs[0]: caffe_in})
                prob = out['prob'][0]
                if options.bin:
                    col_feat.append(out[options.feat_layer_name][0])
                col_prediction.append(prob)
            col_prediction = np.concatenate(col_prediction, axis=2)
            if options.bin:
                col_feat = np.concatenate(col_feat, axis=2)
                feat.append(col_feat)
            prediction.append(col_prediction)
        prob = np.concatenate(prediction, axis=1)
        if options.bin:
            feat = np.concatenate(feat, axis=1)

        if zoom > 1:
            zoom_prob = util.interp_map(
                prob, zoom, image_size[1], image_size[0])
        else:
            zoom_prob = prob[:, :image_size[0], :image_size[1]]
        prediction = np.argmax(zoom_prob.transpose([1, 2, 0]), axis=2)
        if options.bin:
            out_path = join(options.feat_dir,
                            splitext(image_names[i])[0] + '.bin')
            print('Writing', out_path)
            write_array(out_path, feat.astype(np.float32))
            feat_list.append(out_path)
        out_path = join(options.result_dir,
                        splitext(image_names[i])[0] + '.png')
        print('Writing', out_path)
        cv2.imwrite(out_path, prediction)
        result_list.append(out_path)

    print('================================')
    print('All results are generated.')
    print('================================')

    result_list_path = join(options.result_dir, 'results.txt')
    print('Writing', result_list_path)
    with open(result_list_path, 'w') as fp:
        fp.write('\n'.join(result_list))
    if options.bin:
        feat_list_path = join(options.feat_dir, 'feats.txt')
        print('Writing', feat_list_path)
        with open(feat_list_path, 'w') as fp:
            fp.write('\n'.join(feat_list))


def test_bin(options):
    label_margin = 0
    input_zoom = 8
    pad = 0
    if options.up:
        zoom = 1
    else:
        zoom = 8

    if options.gpu >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(options.gpu)
        print('Using GPU ', options.gpu)
    else:
        caffe.set_mode_cpu()
        print('Using CPU')

    net = caffe.Net(options.deploy_net, options.weights, caffe.TEST)

    image_paths = [line.strip() for line in open(options.image_list, 'r')]
    bin_paths = [line.strip() for line in open(options.bin_list, 'r')]
    names = [splitext(split(p)[1])[0] for p in bin_paths]

    assert len(image_paths) == len(bin_paths)

    input_dims = net.blobs['data'].shape
    assert input_dims[0] == 1
    batch_size, num_channels, input_height, input_width = input_dims
    caffe_in = np.zeros(input_dims, dtype=np.float32)

    bin_test_image = read_array(bin_paths[0])
    bin_test_image_shape = bin_test_image.shape
    assert bin_test_image_shape[1] <= input_height and \
        bin_test_image_shape[2] <= input_width, \
        'input_size should be greater than bin image size {} x {}'.format(
            bin_test_image_shape[1], bin_test_image_shape[2])

    result_list = []

    for i in range(len(image_paths)):
        print('Predicting', bin_paths[i])
        image = cv2.imread(image_paths[i])
        image_size = image.shape
        if input_zoom != 1:
            image_rows = image_size[0] // input_zoom + \
                         (1 if image_size[0] % input_zoom != 0 else 0)
            image_cols = image_size[1] // input_zoom + \
                         (1 if image_size[1] % input_zoom != 0 else 0)
        else:
            image_rows = image_size[0]
            image_cols = image_size[1]
        image_bin = read_array(bin_paths[i])
        image_bin = image_bin[:, :image_rows, :image_cols]

        top = label_margin
        bottom = input_height - top - image_rows
        left = label_margin
        right = input_width - left - image_cols

        for j in range(num_channels):
            if pad == 1:
                caffe_in[0][j] = cv2.copyMakeBorder(
                    image_bin[j], top, bottom, left, right,
                    cv2.BORDER_REFLECT_101)
            elif pad == 0:
                caffe_in[0][j] = cv2.copyMakeBorder(
                    image_bin[j], top, bottom, left, right,
                    cv2.BORDER_CONSTANT)
        out = net.forward_all(**{net.inputs[0]: caffe_in})
        prob = out['prob'][0]
        if zoom > 1:
            prob = util.interp_map(prob, zoom, image_size[1], image_size[0])
        else:
            prob = prob[:, :image_size[0], :image_size[1]]
        prediction = np.argmax(prob.transpose([1, 2, 0]), axis=2)
        out_path = join(options.result_dir, names[i] + '.png')
        print('Writing', out_path)
        cv2.imwrite(out_path, prediction)
        result_list.append(out_path)

    print('================================')
    print('All results are generated.')
    print('================================')

    result_list_path = join(options.result_dir, 'results.txt')
    print('Writing', result_list_path)
    with open(result_list_path, 'w') as fp:
        fp.write('\n'.join(result_list))


def test(options):
    if options.model == 'context':
        test_bin(options)
    else:
        test_image(options)


def process_options(options):
    assert exists(options.image_list), options.image_list + ' does not exist'
    assert exists(options.weights), options.weights + ' does not exist'
    assert options.model != 'context' or exists(options.bin_list), \
        options.bin_list + ' does not exist'

    if options.model == 'frontend':
        options.model += '_vgg'

    work_dir = options.work_dir
    model = options.model
    options.deploy_net = join(work_dir, model + '_deploy.txt')
    options.result_dir = join(work_dir, 'results', options.sub_dir, model)
    options.feat_dir = join(work_dir, 'bin', options.sub_dir, model)

    if options.input_size is None:
        options.input_size = [80, 80] if options.model == 'context' \
            else [900, 900]
    elif len(options.input_size) == 1:
        options.input_size.append(options.input_size[0])

    if not exists(work_dir):
        print('Creating working directory', work_dir)
        os.makedirs(work_dir)
    if not exists(options.result_dir):
        print('Creating', options.result_dir)
        os.makedirs(options.result_dir)
    if options.bin and not exists(options.feat_dir):
        print('Creating', options.feat_dir)
        os.makedirs(options.feat_dir)

    return options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', nargs='?',
                        choices=['frontend', 'context', 'joint'])
    parser.add_argument('--work_dir', default='training/',
                        help='Working dir for training.')
    parser.add_argument('--sub_dir', default='',
                        help='Subdirectory to store the model testing results. '
                             'For example, if it is set to "val", the testing '
                             'results will be saved in <work_dir>/results/val/ '
                             'folder. By default, the results are saved in '
                             '<work_dir>/results/ directly.')
    parser.add_argument('--image_list', required=True,
                        help='List of images to test on. This is required '
                             'for context module to deal with variable image '
                             'size.')
    parser.add_argument('--bin_list', help='The input for context module')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--bin', action='store_true',
                        help='Turn on to output the features of a '
                             'layer. It can be useful to generate input for '
                             'context module.')
    parser.add_argument('--feat_layer_name', default=None,
                        help='Extract the response maps from this layer. '
                             'It is usually the penultimate layer. '
                             'Usually, default is good.')
    parser.add_argument('--mean', nargs='*', default=[102.93, 111.36, 116.52], type=float,
                        help='Mean pixel value (BGR) for the dataset.\n'
                             'Default is the mean pixel of PASCAL dataset.')
    parser.add_argument('--input_size', nargs='*', type=int,
                        help='The input image size for deploy network.')
    parser.add_argument('--classes', type=int, required=True,
                        help='Number of categories in the data')
    parser.add_argument('--up', action='store_true',
                        help='If true, upsample the final feature map '
                             'before calculating the loss or accuracy')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU for testing. If it is less than 0, '
                             'CPU is used instead.')
    parser.add_argument('--layers', type=int, default=8,
                        help='Used for training context module.\n'
                             'Number of layers in the context module.')

    options = process_options(parser.parse_args())
    deploy_net, feat_name = make_deploy(options)
    if options.feat_layer_name is None:
        options.feat_layer_name = feat_name
    print('Writing', options.deploy_net)
    with open(options.deploy_net, 'w') as fp:
        fp.write(str(deploy_net))
    test(options)


if __name__ == '__main__':
    main()
