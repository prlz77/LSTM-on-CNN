# -*- coding: utf-8 -*-
""" Converts binaryproto file to npy. """

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

import os
import sys
from config import CAFFE_PATH
sys.path.insert(0, os.path.join(CAFFE_PATH, 'python'))
sys.path.insert(1, os.path.join(CAFFE_PATH, 'tools/extra'))
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import numpy as np

def proto2npy(path):
    """ Loads binaryproto and returns numpy array """
    with open(path) as infile:
        blob = caffe_pb2.BlobProto().FromString(infile.read())

    return caffe.io.blobproto_to_array(blob)

# Main loop
if __name__ == '__main__':
    path = sys.argv[1]
    blob_array = proto2npy(path)
    if len(sys.argv) < 3:
        sys.argv.append(sys.argv[1] + '.npy')
    np.save(sys.argv[2], blob_array[0])
