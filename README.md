# LSTM-on-CNN
Extracts features from a pre-trained CNN and trains a LSTM. 

**Updates**
* Specific test batchsize. (new)
* Label offsets. Max batch-size inference. 
* gen_outputs.py can now read multilabeled files, and ignore missing images.
* Masked input for many-to-one prediction. 
* Early stop. Choosing GPU.
* Added script to save images into h5 file.
* AUC score, better sorting frames in gen_output, saving best outputs.
* Correct and incorrect data examples in this README
* Docker Image.
* Added Classification tests.
* Added Classification task.

## Publications
The code has been used in the following publications:

```
@article{rodriguez2017deep,
  title={Deep Pain: Exploiting Long Short-Term Memory Networks for Facial Expression Classification},
  author={Rodriguez, Pau and Cucurull, Guillem and Gonz{\`a}lez, Jordi and Gonfaus, Josep M and Nasrollahi, Kamal and Moeslund, Thomas B and Roca, F Xavier},
  journal={IEEE Transactions on Cybernetics},
  year={2017},
  publisher={IEEE}
}
@inproceedings{bellantonio2016spatio,
  title={Spatio-Temporal Pain Recognition in CNN-based Super-Resolved Facial Images},
  author={Bellantonio, Marco and Haque, Mohammad A and Rodriguez, Pau and Nasrollahi, Kamal and Telve, Taisi and Escarela, Sergio and Gonzalez, Jordi and Moeslund, Thomas B and Rasti, Pejman and Anbarjafari, Gholamreza},
  booktitle={International Conference on Pattern Recognition (icpr)},
  year={2016},
  organization={Springer}
}
```

## Installation and dependences
1. Install CUDA and CUDNN if possible.
2. Install and compile caffe https://github.com/BVLC/caffe
3. `pip install h5py numpy`
4. Install opencv and python-opencv
5. Install torch http://torch.ch/
6. Install `torch-hdf5` (https://github.com/deepmind/torch-hdf5/blob/master/doc/usage.md)
7. `luarocks install nn; luarocks install cunn; luarocks install rnn; luarocks install optim; luarocks install gnuplot; luarocks install dp; luarocks install class`
8. `git clone this repository`
9. Set your caffe installation path in `config.py`

## Docker Image
``docker pull prlz77/lstm-on-cnn``

## Automatic Script
For automatic generation of the features and training the LSTM run:

```bash
./scripts/train.sh
```
### Data Format
* Images in a OpenCV compatible format (jpg, png, etc.)
* A file with the ordered list of *train* frames. `./path/to/frame_x.jpg label sequence_number\n...`
* A file with the ordered list of *validation* frames. `./path/to/frame_x.jpg label sequence_number\n...`

All the frames from the same video sequence must have the same sequence number. The order is important, e.g:

Wrong sequence numbering:
```
path label 1
path label 2
path label 3
...
path label 1
path label 2
path label 3
```
Correct way:
```
path label 1 (frame 1 of seq 1)
path label 1 (frame 2 of seq 1)
path label 1 (...)
....
path label 2 (frame 1 of seq 2)
path label 2 (frame 2 of seq 2)
path label 2 (...)
...
3
3
3
...
```
Since the LSTM is fed with CNN feature maps, a pre-trained model is needed. For generic baselines I would recommend any of the famous caffemodels in https://github.com/BVLC/caffe/wiki/Model-Zoo. For better results, fine-tune one of them with the specific dataset. Before using it with this code.

### Configuration
The script can be configured from inside:

```bash
# DATASET PATH
DEPLOY='smth.prototxt'
CAFFEMODEL='smth.caffemodel'
DATAROOT='root/to/images/'
TRAIN_LIST='train/list.txt' # ex: ./class/0345435.jpg label seq_num\n... or ./0445342.jpg label seq_num\n, etc.
VAL_LIST='val/list.txt' # ex: ./class/0345435.jpg label seq_num\n... or ./0445342.jpg label seq_num\n, etc.

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
TASK='regress' # task should be in {regress, classify}

# Other
SNAPSHOT_EVERY=10 #number of epochs to save current model. Set 0 for never.
PLOT=1 #plots the regression outputs and targets in real time during learning. Set n to plot every n epochs.
LOG='' #for specific log file. default is ./logs/current_datetime.log. Usage LOG='--logPath <path>'

# FLAGS
# Can use any in {--sort, --cpuonly, --standarize, --verbose} with spaces inbetween
FLAGS='--standarize' # use '--sort' in case the image lists do not have ordered frames
```

## Notes on Classification task
Remember lua (and torch) indexes vectors starting from 1 instead of 0. Thus, labels should be an integer from 1 to #labels.
By default the number of labels is ``max(labels)`` but can be manually set using ``--nlabels``

## Manual usage
* `gen_outputs.py` receives a caffemodel, the images and the listfiles and creates an hdf5 file with the outputs.
* `LSTM.lua` trains a LSTM with a training and validation hdf5 datasets with the following fields:
  - `outputs`: `NxCxHxW` tensor of N images with `C` channels and spatial dims `HxW`.
  - `labels`: `NxC` tensor of `N` labels of `C` dimensionality.
  - `seq_numbers`: tensor of `N` numbers correponding to the video sequence from which the frame was extracted.

Use the `--testonly` and `--saveOutputs` options of `LSTM.lua` in order to extract a HDF5 with the output of the network, and the `--load` option to reload a saved model.

## How to save images into h5
Given a train.txt and a test.txt file with: im_path label seq_num\n...

If we want to save the images into a database after subtracting the mean, resizing to 100px X 100px, and standarizing the labels, we can call:

```bash
# Save train.
python images2h5.py images/root/folder train.txt --output train.h5 \
--size 100 --standarize --subtract_mean --output train.h5
# Save test. We indicate the train.h5 so that the train mean, std, etc. are reused and we do not overfit.
python images2h5.py images/root/folder test.txt --output test.h5 \
--size 100 --standarize --subtract_mean --output test.h5 --with_train train.h5 
```

## Tests
I added some tests to verify it correctly works. You can run them as well to check everything is fine.

## Troubleshooting
Please contact me if any problem `pau.rodriguez at uab.cat`


