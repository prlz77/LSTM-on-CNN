#!/usr/bin/env python
""" Generates the outputs of an arbitrary CNN layer. """

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"


import sys
import os
sys.path.insert(0, os.path.realpath(__file__))
from config import CAFFE_PATH
sys.path.insert(0, os.path.join(CAFFE_PATH, 'python'))
import caffe
import argparse
import h5py
import numpy as np


# CMD Options
parser = argparse.ArgumentParser(description='Generates the outputs of an arbitrary CNN layer.')
parser.add_argument('model', type=str, help='The model deploy file.')
parser.add_argument('weights', type=str, help='The model weights file.')
parser.add_argument('layer', type=str, nargs='+', help='The target layer(s).')
parser.add_argument('--output', type=str, help='The output file.', default='output.h5')
parser.add_argument('--flist', nargs=2, type=str, help='The base folder and the file list of the images.', default=None)
parser.add_argument('--dataset', type=str, help='The lmdb dataset.', default=None)
parser.add_argument('--sort', type=bool, default=False)
parser.add_argument('--mean', type=int, nargs=3, default=None, help='Pixel mean (3 values)')
parser.add_argument('--mean_file', type=str, default=None, help='Per-pixel mean in bgr')
parser.add_argument('--scale', type=int, default=None, help='Scale value.')
parser.add_argument('--swap', type=bool, default=True, help='BGR <-> RGB.')
parser.add_argument('--cpuonly', action='store_false', description='CPU-Only flag.')
args = parser.parse_args()

if not args.cpuonly:
    caffe.set_mode_gpu()

net = caffe.Net(args.model, args.weights,
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
if args.mean != None:
    transformer.set_mean('data', np.array(args.mean))#np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
elif args.mean_file != None:
    if 'npy' in args.mean_file:
        mean = np.load(args.mean_file)
        transformer.set_mean('data', mean)
    else:
        from binaryproto2npy import proto2npy
        mean = proto2npy(args.mean_file)[0]
        transformer.set_mean('data', mean)
if args.scale:
    transformer.set_raw_scale('data', args.scale)  # the reference model operates on images in [0,255] range instead of [0,1]
if args.swap:
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

LMDB=args.dataset

from lmdb_visualizer import get_lmdb_iterator



import Image
import StringIO
import caffe.proto.caffe_pb2


labels = []
outputs = []
seq_numbers_set = []
seq_nums = []
if args.flist != None:
    from cv2 import imread
    with open(args.flist[1], 'r') as infile:
        flist = infile.readlines()
        if args.sort:
            flist.sort()
    for layer in args.layer:
        outputs.append(h5py.File(args.output + '_' + layer.replace('/','_') + '.h5', 'w'))
        outputs[-1].create_dataset('outputs', tuple([len(flist)] + list(net.blobs[layer].data.shape)), dtype='float32')
        outputs[-1].create_dataset('labels', (len(flist), 1), dtype='int32')
        outputs[-1].create_dataset('seq_number', (len(flist), 1), dtype='int32')
    for i,line in enumerate(flist):
        spline = line.replace('\n', '').split(" ")
        img = imread(os.path.join(args.flist[0], spline[0]))
        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        out = net.forward()
        ## REMOVE
        seq_number = spline[0].split("/")[3]
        if seq_number not in seq_numbers_set:
            seq_numbers_set.append(seq_number)
            print 'Current seeq number: ', seq_numbers_set.index(seq_number)
        seq_nums.append(seq_numbers_set.index(seq_number))
        ## END REMOVE
        labels.append(float(spline[1]))

        for index, layer in enumerate(args.layer):
            outputs[index]['outputs'][i,...] = net.blobs[layer].data[...]
            outputs[index]['labels'][i,...] = labels[-1]
            outputs[index]['seq_number'][i,...] = seq_nums[-1]
        if i % 1000 == 0:
            print "Processing image ", i
elif args.dataset != None:
    iterator = get_lmdb_iterator(LMDB)

    for index, it in enumerate(iterator):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(iterator.value())
        label = datum.label
        img = np.asarray(Image.open(StringIO.StringIO(datum.data)))


        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        out = net.forward()
        outputs.append(net.blobs[args.layer].data.copy())
        if index % 1000 == 0:
            print "Processing image ", index

else:
    raise Exception('need a dataset')


for index, layer in enumerate(args.layer):
    outputs[index].close()

# f = h5py.File(args.output, 'w')
#
# f.create_dataset('outputs', data=np.asarray(outputs), compression="gzip")
# f.create_dataset('labels', data=np.asarray(labels))
# f.create_dataset('seq_number', data=np.asarray(seq_nums))
# f.close()