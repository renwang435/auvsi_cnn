#Appends a label_byte to input images based on the shape contained within the filename
#This is needed for both the shape recognition (CNN) and alphanumeric recognition (deep FFN) modules to train
#Writes all files to a data-dir

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--img_dir', type=str, default='./raw-imgs/',
                    help='Path to the images directory.')

parser.add_argument('--data_dir', type=str, default='./training-batches-bin/',
                    help='Path to the output data directory.')

FLAGS = parser.parse_args()

def gen_img_bin():
    """
    Function to create initialized Variable with weight decay
    Args:
        img_dir: Directory of images in JPEG format
        data_dir: Directory where training/eval data will be written to
    Return:
        None
    """
    pass

def preproc():
    pass