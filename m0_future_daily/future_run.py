import argparse
import os
import sys
import time
import numpy as np
from six.moves import xrange
from m0_future_daily.data_process import read_data
from m0_future_daily.future_model import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Basic model parameters as external flags.
FLAGS = None


def placeholder_inputs(batch_size):
    """
    Args:
        holder_size: The size will be baked into both placeholders.
    Returns:
        features_placeholder: Features placeholder.
        labels_placeholder: Labels placeholder.
    """
    features_placeholder = tf.placeholder(tf.float32, shape=[None, NUM_FEATURES])
    labels_placeholder = tf.placeholder(tf.int32, shape=[None])
    return features_placeholder, labels_placeholder


def next_batch(num, features, labels):
    """
    Args:
        num: Size of next batch
        features: Features of data
        labels: Labels of data
    Returns:
        next_features: features of next batch
        next_labels: labels of next batch
    """
    idx = np.arange(0, len(features))
    np.random.shuffle(idx)
    idx = idx[0:num]
    feature_shuffle = [features[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    next_features = np.asarray(feature_shuffle)
    next_labels = np.asarray(labels_shuffle)
    return next_features, next_labels


def fill_feed_dict(features, labels, features_pl, labels_pl):
    features_feed, labels_feed = next_batch(FLAGS.batch_size, features, labels)
    feed_dict = {features_pl: features_feed, labels_pl: labels_feed}
    return feed_dict


def do_eval(sess, eval_correct, features_placeholder, labels_placeholder, feature_eval, label_eval):
    """
    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Op that returns the number of correct predictions.
        features_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        feature_eval: The set of features to evaluate
        label_eval: The set of labels to evaluate
    """
    '''
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = len(label_eval) // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(feature_eval, label_eval, features_placeholder, labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))
    '''

    feed_dict = {features_placeholder: feature_eval, labels_placeholder: label_eval}
    true_count = sess.run(eval_correct, feed_dict=feed_dict)
    num_examples = len(label_eval)
    precision = float(true_count) / num_examples
    print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))


def output_prediction(sess, logits, features_placeholder, test_feature):
    feed_dict = {features_placeholder: test_feature}
    prediction = tf.argmax(logits, 1)
    best = sess.run([prediction], feed_dict)
    print(best)


def run_training():
    # Get the sets of features and labels for training and test
    train_features, train_labels, test_features, test_labels = read_data()

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the features and labels.
        features_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

        # Build a Graph that computes predictions from the inference model.
        logits = inference(features_placeholder, FLAGS.hidden1, FLAGS.hidden2)

        # Add to the Graph the Ops for loss calculation.
        loss_op = loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(loss_op, FLAGS.learning_rate)

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
            # feed_dict = fill_feed_dict(train_features, train_labels, features_placeholder, labels_placeholder)
            feed_dict = {features_placeholder: train_features, labels_placeholder: train_labels}

            # Run one step of the model.
            _, loss_value = sess.run([train_op, loss_op], feed_dict=feed_dict)

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
                do_eval(sess, eval_correct, features_placeholder, labels_placeholder, train_features, train_labels)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess, eval_correct, features_placeholder, labels_placeholder, test_features, test_labels)

        output_prediction(sess, logits, features_placeholder, test_features)


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='Initial learning rate.')
    parser.add_argument('--max_steps', type=int, default=30000, help='Number of steps to run trainer.')
    parser.add_argument('--hidden1', type=int, default=128, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=64, help='Number of units in hidden layer 2.')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size.')
    parser.add_argument('--log_dir', type=str, default='.\savedir/low_level', help='Directory to put the log data.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
