from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import CNN_Arch
import Train_Preproc

parser = Train_Preproc.parser

parser.add_argument('--train_dir', type=str, default='./CNN_Train',
                    help='Directory where to write event logs and checkpoint.')

parser.add_argument('--max_steps', type=int, default=1000000,
                    help='Number of batches to run.')

parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')

parser.add_argument('--log_frequency', type=int, default=10,
                    help='How often to log results to the console.')


def train():
  """Train the model for specified a number of steps."""

  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = CNN_Arch.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = CNN_Arch.inference(images)

    # Calculate loss.
    loss = CNN_Arch.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = CNN_Arch.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument

  #Need to input the data after Train_Preproc runs
  Train_Preproc.gen_img_bin()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  FLAGS = parser.parse_args()
  tf.app.run()

