import os
import tensorflow as tf
import numpy as np
import cv2

import CNN_Arch

#Relative Path to meta file
#Relative Path to image
#Relative Path to pre-trained weights

def restore_vars(saver, sess, chkpt_dir):
    """ Restore saved net, global score and step, and epsilons OR
    create checkpoint directory for later storage. """
    sess.run(tf.global_variables_initializer())

    checkpoint_dir = chkpt_dir

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

if __name__ == '__main__':
    to_restore = True
    with tf.Graph().as_default():

        # Gets the image
        images = cv2.imread('./test2.jpg')

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
                restored = restore_vars(saver, sess, './CNN_Train')
                to_restore = False

            _, top_k_pred = tf.nn.top_k(logits, k=5)
            top_indices = sess.run([top_k_pred])

            print("Predicted ", top_indices[0], " for your input image.")
            # logit_val = sess.run(logits)
            # print(logit_val)




