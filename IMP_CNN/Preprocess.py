#Appends a label_byte to input images based on the shape contained within the filename
#This is needed for both the shape recognition (CNN) and alphanumeric recognition (deep FFN) modules to train
#Writes all files to a data-dir

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import scipy
import scipy.misc
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--preproc', type=bool, default=False,
                    help='Whether or not to pre-process images.')

parser.add_argument('--img_dir', type=str, default='./raw-imgs/',
                    help='Path to the images directory.')

parser.add_argument('--data_dir', type=str, default='./training-batches-bin/',
                    help='Path to the output data directory.')

path = "./test_images/"
dirs = os.listdir(path)

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

def global_contrast_normalization(filename, s, lmda, epsilon):
    X = np.array(Image.open(filename))

    # replacement for the loop
    X_average = np.mean(X)
    print('Mean: ', X_average)
    X = X - X_average

    # `su` is here the mean, instead of the sum
    contrast = np.sqrt(lmda + np.mean(X**2))

    X = s * X / max(contrast, epsilon)

    # scipy can handle it
    scipy.misc.imsave(filename, X)

def resize_normalize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((32,32), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=100)
            #global_contrast_normalization(f + '.jpg', 1, 10, 0.000000001)



resize_normalize()
