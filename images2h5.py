import argparse
import h5py
import cv2
import os
import numpy as np

parser = argparse.ArgumentParser(description="Reads a list of images, labels and sequences and outputs it into an h5")
parser.add_argument('flist', type=str, nargs=2, help="root path and file list with image<space>label<space>seq_num\n")
parser.add_argument('--subtract_mean', action='store_true', help='standarize data')
parser.add_argument('--output', type=str, help='output filename')
parser.add_argument('--size', type=int, default=0, help='resize images to size')
parser.add_argument('--standarize', action='store_true', help='whether to standarize the labels (useful for regression)')
parser.add_argument('--with_train', type=str, default='', help='standarize/mean test given the train.h5')

args = parser.parse_args()

print 'Reading flist'
with open(args.flist[1], 'r') as infile:
    data = infile.readlines()

if args.size == 0:
    sample = cv2.imread(os.path.join(args.flist[0], data[0].split(' ')[0]))
    args.size = sample.shape[0] # assuming square images
    resize = False
else:
    resize = True

out = h5py.File(args.output, 'w')

train = None
if os.path.exists(args.with_train):
    train = h5py.File(args.with_train, 'r')

out.create_dataset('outputs', shape=(len(data), 3, args.size, args.size), dtype=float)
out.create_dataset('labels', shape=(len(data), 1), dtype=float)
out.create_dataset('seq_number', shape=(len(data), ), dtype='int32')

if train is None:
    mean_im = np.zeros((3,1,1))
else:
    mean_im = train['mean'][...]

print 'Reading and pre-processing images'
for i,d in enumerate(data):
    dir, label, seq = d.replace('\n', '').split(' ')
    im = cv2.imread(os.path.join(args.flist[0], dir))
    if resize:
        im = cv2.resize(im, dsize=(args.size, args.size), interpolation=cv2.INTER_CUBIC)
    im = im.transpose(2,0,1).astype('float')
    if args.subtract_mean and train is None:
        mean_im += im.mean(axis=(1,2))[:,None,None]
    elif train is not None:
        im -= mean_im
    out['outputs'][i,...] = im
    out['labels'][i,0] = float(label)
    out['seq_number'][i] = int(seq)
    if i % 100 == 0:
        print 100*i/float(len(data)), '%'
print '100 %'
if args.subtract_mean and train is None:
    mean_im /= len(data)
    for i in xrange(len(data)):
        out['outputs'][i,...] -= mean_im

out['mean'] = mean_im

if train is not None:
    label_mean = train['label_mean'].value
    label_std = train['label_std'].value
    out['label_mean'] = label_mean
    out['label_std'] = label_std
    out['labels'][:] -= label_mean
    out['labels'][:] /= label_std
elif args.standarize:
    out['label_mean'] = out['labels'][:].mean()
    out['label_std'] = out['labels'][:].std()
    out['labels'][:] = out['labels'][:] - out['label_mean']
    out['labels'][:] = out['labels'][:] / out['label_std']
else:
    out['label_mean'] = 0.
    out['label_std'] = 1.

out.close()
if train is not None:
    train.close()

print 'Done'