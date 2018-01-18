from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange

import tensorflow as tf

#Define input image size in px (64x64)
#IMG_SIZE = 64
IMG_SIZE = 24

#Global constants based on dataset
#Will likely have to change number of shape classes
#Epoch constants used in CNN_Train and CNN_Eval
NUM_CLASSES = 10
#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 100
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 20
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def read_imgs(filename_queue):
    """
    Reads and parse images

    Args:
        filename_queue: A queue of strings with the filenames of images to read from

    Returns:
        An object representing a single example:
            height: # of rows
            width: # of cols
            depth: # of color channels
            key: scalar string Tensor describing the filename for this example
            label: int32 Tensor with the label in the range 0-9
            uint8image: a [height, width, depth] uint8 Tensor with the image data
    """

    class IMGObj(object):
        pass

    result = IMGObj()

    #Dimensions of the image
    label_bytes = 1     #Note we will have to expand this if more than 10 classes
    result.height = 32
    result.width = 32
    result.depth = 3

    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)


    #Convert the first byte to the label
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    #The remaining bytes after the label correspond to the image --> reshape
    #from [depth * height * width] to [depth, height, width]
    depth = tf.reshape(
        tf.strided_slice(
            record_bytes, [label_bytes], [label_bytes + image_bytes]), [result.depth, result.height, result.width]
        )

    #Convert [depth, height, width] to [height, width, depth]
    result.uint8image = tf.transpose(depth, [1, 2, 0])

    return result

def _gen_img_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):

    """
    Construct a queued batch of images and labels

    Args:
    image: 3D Tensor of [height, width, 1], type float32
    label: 1D Tensor of type int32
    min_queue_examples: int32, minimum number of samples to retain in the queue which provides batches of examples
    batch_size: Number of images per batch
    shuffle: Boolean to indicate whether or not we shuffle the queue

    Returns:
        images: 4D tensor of [batch_size, height, width, 1]
        labels: 1D tensor of [batch_size}
    """

    #Create a queue which shuffles the examples, reads batch_size
    #images + labels from the examples queue
    num_preproc_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preproc_threads,
            capacity=min_queue_examples + batch_size,
            min_after_dequeue=min_queue_examples
        )
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preproc_threads,
            capacity=min_queue_examples + batch_size
        )

        #Display training images in TensorBoard visualizer
        tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])

def distorted_inputs(data_dir, batch_size):

    """
    Distorts inputs for additional training examples

    Args:
    data_dir: Path to images directory
    batch_size: NUmber of images per batch

    Returns:
        images: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1]
        labels: 1D tensor of [batch_size]
    """

    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('File Not Found: ' + f)

    #Create queue which we populate with filenames to read
    filename_queue = tf.train.string_input_producer(filenames)

    #Read examples from files in filename queue
    read_input = read_imgs(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMG_SIZE
    width = IMG_SIZE

    #Apply transformations to distort images

    #Randomly crop a [height, width] section of the image
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    #Randomly flip the image horizontally
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    #Randomly flip the image vertically
    distorted_image = tf.image.random_flip_up_down(distorted_image)

    #Randomly alter brightness and contrast
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    #Subtract mean and divide by variance of pixels (local contrast normalization?)
    float_image = tf.image.per_image_standardization(distorted_image)

    #Set shapes of tensors
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    #Ensure random shuffling of examples has good mixing properties
    min_fraction_of_examples_in_queue = 0.25
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print('Filling queue with %d images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    #Generate batch of images and labels via a queue
    return _gen_img_and_label_batch(float_image, read_input.label,
                                    min_queue_examples, batch_size,
                                    shuffle=True)

def inputs(eval_data, data_dir, batch_size):

    """
    Construct inputs for model evaluation using the Reader ops

    Args:
    eval_data: bool, indicating whether training or testing set should be used
    data_dir: Path to the images directory
    batch_size: Number of images per batch

    Returns:
        images: 4D tensor of [batch_size, IMG_SIZE, IMG_SIZE, 1]
        labels: 1D tensor of [batch_size]
    """

    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                     for i in xrange(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('FIle Not Found: ' + f)

    #Create queue which we populate with image filenames to read
    filename_queue = tf.train.string_input_producer(filenames)

    #Read images from files in filename queue
    read_input = read_imgs(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMG_SIZE
    width = IMG_SIZE

    #Subtract mean and divide by variance of pixels
    float_image = tf.image.per_image_standardization(reshaped_image)

    #Set shapes of tensors
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    #Ensure that random shuffling has good mixing properties
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

    #Generate batch of images + labels via a queue
    return _gen_img_and_label_batch(float_image, read_input.label,
                                    min_queue_examples, batch_size,
                                    shuffle=False)