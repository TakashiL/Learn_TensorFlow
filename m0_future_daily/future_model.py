import math
import numpy as np
import csv
import tensorflow as tf

# data files
TRAIN_FILE = "train_data.csv"
TEST_FILE = "test_data.csv"

# parameter
NUM_CLASSES = 2
NUM_FEATURES = 6


def read_data(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data_list = list(reader)
        features = []
        labels = []
        for item in data_list:
            item_feature = list(map(float, item[:6]))
            item_feature = list(map(int, item_feature))
            item_label = int(item[6])
            features.append(item_feature)
            labels.append(item_label)

    features = np.asarray(features)
    labels = np.asarray(labels)
    return features, labels


def inference(features, hidden1_units, hidden2_units):
    """
    Args:
        features: features placeholder, from inputs().
        hidden1_units: Size of the first hidden layer.
        hidden2_units: Size of the second hidden layer.
    Returns:
        softmax_linear: Output tensor with the computed logits.
    """
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, hidden1_units], stddev=1.0 / math.sqrt(float(NUM_FEATURES))), name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(features, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units], stddev=1.0 / math.sqrt(float(hidden1_units))), name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([hidden2_units, NUM_CLASSES], stddev=1.0 / math.sqrt(float(hidden2_units))), name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits


def lossfn(logits, labels):
    """
    Args
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].
    Returns:
        loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
    """
    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.
    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    """
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the range [0, NUM_CLASSES).
    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size) that were predicted correctly.
    """
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def placeholder_inputs(batch_size):
    """
    Args:
        batch_size: The batch size will be baked into both placeholders.
    Returns:
        features_placeholder: Features placeholder.
        labels_placeholder: Labels placeholder.
    """
    features_placeholder = tf.placeholder(tf.float32, shape=(batch_size, NUM_FEATURES))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
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
    idx = idx[:num]
    feature_shuffle = [features[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    next_features = np.asarray(feature_shuffle)
    next_labels = np.asarray(labels_shuffle)
    return next_features, next_labels


def fill_feed_dict(feature_set, label_set, features_pl, labels_pl, batch_size):
    """
    Args:
        feature_set: The set of features
        label_set: The set of labels
        features_pl: The features placeholder, from placeholder_inputs().
        labels_pl: The labels placeholder, from placeholder_inputs().
    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next `batch size` examples.
    features_feed, labels_feed = next_batch(batch_size, feature_set, label_set)
    feed_dict = {features_pl: features_feed, labels_pl: labels_feed}
    return feed_dict


def do_eval(sess, eval_correct, features_placeholder, labels_placeholder, feature_eval, label_eval, batch_size):
    """
    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Op that returns the number of correct predictions.
        features_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        feature_eval: The set of features to evaluate
        label_eval: The set of labels to evaluate
        batch_size: Size of batch
    """
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = len(feature_eval) // batch_size
    num_examples = steps_per_epoch * batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(feature_eval, label_eval, features_placeholder, labels_placeholder, batch_size)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))


