import argparse
import os
import sys
import time
import tensorflow as tf
from m0_future_daily.future_model import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Basic model parameters as external flags.
FLAGS = None


def run_training():
    # Get the sets of features and labels for training and test
    train_features, train_labels = read_data(TRAIN_FILE)
    test_features, test_labels = read_data(TEST_FILE)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the features and labels.
        features_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

        # Build a Graph that computes predictions from the inference model.
        logits = inference(features_placeholder, FLAGS.hidden1, FLAGS.hidden2)

        # Add to the Graph the Ops for loss calculation.
        loss = lossfn(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(loss, FLAGS.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = evaluation(logits, labels_placeholder)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        # Start the training loop.
        for step in range(FLAGS.max_steps):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of features and labels for this particular training step.
            feed_dict = fill_feed_dict(train_features, train_labels, features_placeholder, labels_placeholder, FLAGS.batch_size)

            # Run one step of the model.
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess, eval_correct, features_placeholder, labels_placeholder, train_features, train_labels, FLAGS.batch_size)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess, eval_correct, features_placeholder, labels_placeholder, test_features, test_labels, FLAGS.batch_size)


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--max_steps', type=int, default=30000, help='Number of steps to run trainer.')
    parser.add_argument('--hidden1', type=int, default=20, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=20, help='Number of units in hidden layer 2.')
    parser.add_argument('--batch_size', type=int, default=143, help='Batch size.')
    parser.add_argument('--log_dir', type=str, default='/tmp/save', help='Directory to put the log data.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
