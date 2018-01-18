#Appends a label_byte to input images based on the shape contained within the filename
#This is needed for both the shape recognition (CNN) and alphanumeric recognition (deep FFN) modules to train
#Writes all files to a data-dir

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import argparse
import os
import re
import sys
import tarfile
from six.moves import urllib

import CNN_Arch

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='/tmp/img_data',
                    help='Path to the images directory.')

#Takes as input a directory of training images, converts images to the CIFAR10 binary format


def preproc():
    pass