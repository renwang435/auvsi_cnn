import os
import argparse
import tensorflow as tf
import numpy as np
import cv2

import CNN_Arch

parser = argparse.ArgumentParser()

parser.add_argument('--meta_path', type=str, default='./batches.meta.txt',
                    help='Path to txt file containing a class on each line')

parser.add_argument('--img_path', type=str, default='./test.jpg',
                    help='Path to image to run inference on')

parser.add_argument('--train_dir', type=str, default='./CNN_Train',
                    help='Path to directory with pre-trained weights')

infFLAGS = parser.parse_args()

def restore_vars(saver, sess, chkpt_dir):
    """ Restore saved net, global score and step, and epsilons OR
    create checkpoint directory for later storage. """
    sess.run(tf.global_variables_initializer())

    checkpoint_dir = infFLAGS.train_dir

    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir)
        except OSError:
            pass

    path = tf.train.get_checkpoint_state(checkpoint_dir)
    if path is None:
        return False
    else:
        saver.restore(sess, path.model_checkpoint_path)
        return True

def load_meta():
    d = {}
    with open(infFLAGS.meta_path) as f:
        i = 0
        for line in f:
            d[i] = line
            i = i + 1

    return d

if __name__ == '__main__':
    to_restore = True
    with tf.Graph().as_default():

        # Gets the image
        images = cv2.imread(infFLAGS.img_path)

        images = np.asarray(images, dtype=np.float32)
        images = tf.convert_to_tensor(images / 255.0)
        images = tf.image.resize_image_with_crop_or_pad(images, 24, 24)
        images = tf.reshape(images, [1, 24, 24, 3])
        images = tf.cast(images, tf.float32)

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if to_restore:
                logits = CNN_Arch.inference(images)
            else:
                scope.reuse_variables()
                logits = CNN_Arch.inference(images)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            if to_restore:
                restored = restore_vars(saver, sess, infFLAGS.train_dir)
                to_restore = False

            _, top_k_pred = tf.nn.top_k(logits, k=5)
            top_indices = sess.run([top_k_pred])
            top_pred = top_indices[0][0][0]
            dict = load_meta()

            print("Predicted ", dict[top_pred], " for your input image.")
            #print(infFLAGS.train_dir)




