# -*- coding: utf-8 -*-
""" Generates the outputs of an arbitrary CNN layer. """

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

import argparse

# CMD Options
parser = argparse.ArgumentParser(description="""Generates the outputs of an arbitrary CNN layer
accepts either a LMDB dataset or a listfile of images.""")
parser.add_argument('model', type=str, help='The model deploy file.')
parser.add_argument('weights', type=str, help='The model weights file.')
parser.add_argument('layer', type=str, nargs='+', help='The target layer(s).')
parser.add_argument('--output', type=str, help='The output file.', default='output.h5')
parser.add_argument('--flist', nargs=2, type=str, help='The base folder and the file list of the images.', default=None)
parser.add_argument('--label_names', nargs='+', type=str, default=['labels'], help='specific label names, accepts more than one label')
parser.add_argument('--dataset', type=str, help='The lmdb dataset.', default=None)
parser.add_argument('--sort', action='store_true', help="Whether images should be sorted")
parser.add_argument('--mean', type=float, nargs=3, default=None, help='Pixel mean (3 bgr values)')
parser.add_argument('--mean_file', type=str, default=None, help='Per-pixel mean in bgr')
parser.add_argument('--scale', type=int, default=None, help='Scale value.')
parser.add_argument('--swap', action='store_true', help='BGR <-> RGB. If using --flist, images are loaded in BGR by default.')
parser.add_argument('--cpuonly', action='store_true', help='CPU-Only flag.')
parser.add_argument('--standarize', action='store_true', help="whether to standarize the outputs")
parser.add_argument('--standarize_with', type=str, default='', help='get mean and std from another .h5 (recommended for validation)')
parser.add_argument('--save_paths', action='store_true', help="Whether to save the path of the files")
parser.add_argument('--verbose', action='store_true', help='show image paths while being processed')

args = parser.parse_args()

#TODO
if args.standarize and len(args.label_names > 1):
    raise NotImplementedError("This code does not support yet standarizing multiple labels")

# Move the rest of imports to avoid conflicts with argparse
import sys
import os
sys.path.insert(0, os.path.realpath(__file__))
from config import CAFFE_PATH
sys.path.insert(0, os.path.join(CAFFE_PATH, 'python'))
import caffe
import h5py
import numpy as np

# CPU ONLY
if not args.cpuonly:
    caffe.set_mode_gpu()

# Read Deploy + Weights file
net = caffe.Net(args.model, args.weights,
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# images are assumed to be in format hxwxc
transformer.set_transpose('data', (2,0,1))

# get mean
if args.mean != None:
    transformer.set_mean('data', np.array(args.mean))#np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
elif args.mean_file != None:
    if 'npy' in args.mean_file:
        mean = np.load(args.mean_file)
    else:
        from binaryproto2npy import proto2npy
        mean = proto2npy(args.mean_file)[0]
    mean = mean.mean(1).mean(1)
    transformer.set_mean('data', mean)
if args.scale:
    transformer.set_raw_scale('data', args.scale)  # the reference model operates on images in [0,255] range instead of [0,1]
if args.swap:
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# Auxiliar variables
labels = {}
for label_name in args.label_names:
    labels[label_name] = []
outputs = []
seq_numbers_set = []
seq_nums = []
paths = []

# We received a txt listfile, not lmdb
if args.flist != None:
    from cv2 import imread
    with open(args.flist[1], 'r') as infile:
        flist = infile.readlines()
        if args.sort:
            flist.sort()
            current_seq = int(flist[0].split(' ')[-1].replace('\n', ''))
            counter = 1
            for i,v in enumerate(flist):
                vsplit = v.split(" ")
                seq = int(vsplit[-1].replace('\n', ''))
                if seq != current_seq:
                    current_seq = seq
                    counter += 1
                flist[i] = " ".join(vsplit[:-1]) + " %d\n" %(counter)
    for layer in args.layer:
        outputs.append(h5py.File(args.output + '_' + layer.replace('/','_') + '.h5', 'w'))
        dim = list(net.blobs[layer].data.shape)
        if len(dim) < 3:
            dim = [1,1,np.array(dim).prod()]
        outputs[-1].create_dataset('outputs', tuple([len(flist)] + dim), dtype='float32')
        for label_name in args.label_names:
            outputs[-1].create_dataset(label_name, (len(flist), 1), dtype='float')
        outputs[-1].create_dataset('seq_number', (len(flist),), dtype='int32')
    for i,line in enumerate(flist):
        spline = line.replace('\n', '').split(" ")
        img = imread(os.path.join(args.flist[0], spline[0]).replace('\\', '/'))
        if args.verbose:
            print(os.path.join(args.flist[0], spline[0]))
        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        out = net.forward()
        # Get sequence numbers
        seq_number = int(spline[-1])

        if args.verbose and len(seq_nums) >= 1 and seq_number != seq_nums[-1]:
            print "Current sequence: ", seq_number
        seq_nums.append(seq_number)

        # for debugging
        for idx, label_name in enumerate(args.label_names):
            labels[label_name].append(float(spline[1 + idx]))
        
        if args.save_paths:
            paths.append(line)

        for index, layer in enumerate(args.layer):
            outputs[index]['outputs'][i,...] = net.blobs[layer].data[...]
            for label_name in args.label_names:
                outputs[index][label_name][i,...] = labels[label_name][-1]
            outputs[index]['seq_number'][i] = seq_nums[-1]
        if i % 1000 == 0:
            print "Processing image ", i, " of ", len(flist)
    

    for index, layer in enumerate(args.layer):
        if args.save_paths:
            outputs[index]['paths'] = paths
        if os.path.isfile(args.standarize_with):
            train = h5py.File(args.standarize_with, 'r')
            mean = train['mean'][...]
            std = train['std'][...]
            label_mean = train['label_mean'][...]
            label_std = train['label_std'][...]
        elif args.standarize:
            label_mean = outputs[index]['labels'][...].mean()
            label_std = outputs[index]['labels'][...].std()
            mean = outputs[index]['outputs'][...].mean()
            std = outputs[index]['outputs'][...].std()
        else:
            mean = 0.0
            std = 1.0
    if args.standarize or os.path.isfile(args.standarize_with):
        outputs[index]['labels'][...] -= label_mean
        outputs[index]['labels'][...] /= label_std
        outputs[index]['label_mean'] = label_mean
        outputs[index]['label_std'] = label_std
        outputs[index]['outputs'][...] -= mean
        outputs[index]['outputs'][...] /= std
        outputs[index]['mean'] = mean
        outputs[index]['std'] = std

elif args.dataset != None:
    LMDB = args.dataset
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
