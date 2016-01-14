#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import caffe
import cv2
import numba
import numpy as np
from os.path import dirname, exists, join, splitext
import sys

__author__ = 'Fisher Yu'
__copyright__ = 'Copyright (c) 2016, Fisher Yu'
__email__ = 'i@yf.io'
__license__ = 'MIT'


PALLETE = np.array([
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128]
], dtype=np.uint8)


@numba.jit(nopython=True)
def interp_map(prob, zoom, width, height):
    zoom_prob = np.zeros((prob.shape[0], height, width), dtype=np.float32)
    for c in range(prob.shape[0]):
        for h in range(height):
            for w in range(width):
                r0 = h // zoom
                r1 = r0 + 1
                c0 = w // zoom
                c1 = c0 + 1
                rt = float(h) / zoom - r0
                ct = float(w) / zoom - c0
                v0 = rt * prob[c, r1, c0] + (1 - rt) * prob[c, r0, c0]
                v1 = rt * prob[c, r1, c1] + (1 - rt) * prob[c, r0, c1]
                zoom_prob[c, h, w] = (1 - ct) * v0 + ct * v1
    return zoom_prob


def predict(model_path, pretrained, input_path, output_path):
    net = caffe.Net(model_path, pretrained, caffe.TEST)
    label_margin = 186
    zoom = 8
    mean_pixel = np.array([102.93, 111.36, 116.52], dtype=np.float32)
    input_dims = net.blobs['data'].shape
    batch_size, num_channels, input_height, input_width = input_dims
    caffe_in = np.zeros(input_dims, dtype=np.float32)
    image = cv2.imread(input_path, 1).astype(np.float32) - mean_pixel
    if image.shape[0] > 500 or image.shape[1] > 500:
        resize_ratio = \
            500.0 / (image.shape[0] if image.shape[0] > image.shape[1] else
                     image.shape[1])
        image = cv2.resize(image, None, None, resize_ratio, resize_ratio)
    image_size = image.shape
    margin = [label_margin, input_height - image.shape[0] - label_margin,
              label_margin, input_width - image.shape[1] - label_margin]
    image = cv2.copyMakeBorder(image, margin[0], margin[1], margin[2],
                               margin[3], cv2.BORDER_REFLECT_101)
    caffe_in[0] = image.transpose([2, 0, 1])
    out = net.forward_all(**{net.inputs[0]: caffe_in})
    prob = out['prob'][0]
    zoom_prob = interp_map(prob, zoom, image_size[1], image_size[0])
    prediction = np.argmax(zoom_prob.transpose([1, 2, 0]), axis=2)
    color_image = PALLETE[prediction.ravel()].reshape(image_size)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, color_image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', nargs='?', default='',
                        help='Required path to input image')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID to run CAFFE. If -1, CPU is used, '
                             'which is the default setting.')
    parser.add_argument('--context', type=int, default=0,
                        help='Use context module')
    args = parser.parse_args()
    if args.input_path == '':
        sys.exit('Error: No path to input image')
    if not exists(args.input_path):
        sys.exit("Error: Can't find input image " + args.input_path)
    if args.gpu >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)
        print('Using GPU ', args.gpu)
    else:
        caffe.set_mode_cpu()
        print('Using CPU')
    pretrained = join(dirname(__file__), 'pretrained',
                      'dilated_convolution_context_coco.caffemodel')
    if not exists(pretrained):
        raise sys.exit('Error: Run pretrained/download.sh first to '
                       'download pretrained model weights')
    if args.context:
        suffix = '_context.png'
        model_path = join(dirname(__file__), 'models',
                          'dilated_convolution_context.prototxt')
    else:
        suffix = '_front.png'
        model_path = join(dirname(__file__), 'models',
                          'dilated_convolution_front.prototxt')
    output_path = splitext(args.input_path)[0] + suffix
    predict(model_path, pretrained,
            args.input_path, output_path)


if __name__ == '__main__':
    main()