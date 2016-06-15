#!/bin/bash
# Generates CNN feature maps, saves them to HDF5 and trains LSTM
# Author: Pau Rodríguez López (@prlz77)
# Mail: pau.rodri1 at gmail.com
# Institution: ISELAB at CVC-UAB
# Date: 14/06/2016


# DATASET PATH
DEPLOY='smth.prototxt'
CAFFEMODEL='smth.caffemodel'
DATAROOT='root/to/images/'
TRAIN_LIST='train/list.txt' # ex: ./class/0345435.jpg\n... label seq_num or ./0445342.jpg\n... label seq_num, etc.
VAL_LIST='val/list.txt' # ex: ./class/0345435.jpg\n... label seq_num or ./0445342.jpg\n... label seq_num, etc.

# CONVNET PARAMS
EXTRACT_FROM="fc7" # example from vgg16

# DATASET MEAN
# (IMAGENET mean, change as convenient)
R=123.68 
G=116.779
B=123.68

# LSTM hyperparameters
RECURRENT_DEPTH=1
HIDDEN_LAYER_SIZE=256
RHO=5 # max sequence length. n when doing n-to-1.
BATCHSIZE=32 # should be as big as possible
EPOCHS=100000
DROPOUT_PROB=0

# Other
SNAPSHOT_EVERY=10 #number of epochs to save current model. Set 0 for never.
PLOT=1 #plots the regression outputs and targets in real time during learning. Set n to plot every n epochs.
LOG='' #for specific log file. default is ./logs/current_datetime.log. Usage LOG='--logPath <path>'

# FLAGS
# Can use any in {--sort, --cpuonly, --standarize, --verbose} with spaces inbetween
FLAGS='--standarize' # use '--sort' in case the image lists do not have ordered frames


## START OF SCRIPT ##
mkdir -p outputs

echo "1. Extracting feature maps"
python gen_outputs.py $DEPLOY $CAFFEMODEL $EXTRACT_FROM --output outputs/train --flist $DATAROOT $TRAIN_LIST --mean $B $G $R $FLAGS
python gen_outputs.py $DEPLOY $CAFFEMODEL $EXTRACT_FROM --output outputs/val --flist $DATAROOT $VAL_LIST --mean $B $G $R --standarize_with "outputs/train_"$EXTRACT_FROM".h5" $FLAGS

echo "2. Train LSTM"
if [[ $LOG != '' ]]; then
    mkdir -p $LOG
fi
th LSTM.lua --trainPath "./outputs/train_"$EXTRACT_FROM".h5" --valPath "./outputs/val_"$EXTRACT_FROM".h5" \
            --rho $RHO --hiddenSize $HIDDEN_LAYER_SIZE --depth $RECURRENT_DEPTH --batchSize $BATCHSIZE \
            --dropoutProb $DROPOUT_PROB  --plotRegression $PLOT $LOG --saveEvery $SNAPSHOT_EVERY --maxEpoch $EPOCHS