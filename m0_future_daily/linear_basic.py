import csv
import os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LINEAR_TRAIN = "./lineardata/train_data.csv"
LINEAR_TEST = "./lineardata/test_data.csv"


def read_single_linear_data(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data_list = list(reader)
        features = []
        labels = []
        for item in data_list:
            item_feature = list(map(float, item[:5]))
            item_label = float(item[5])
            features.append(item_feature)
            labels.append(item_label)

    features = np.asarray(features)
    labels = np.asarray(labels)
    return features, labels


def read_linear_data():
    train_features, train_labels = read_single_linear_data(LINEAR_TRAIN)
    test_features, test_labels = read_single_linear_data(LINEAR_TEST)
    return train_features, train_labels, test_features, test_labels

learning_rate = 1e-8
epochs = 1000
points = [[], []]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

train_features, train_labels, test_features, test_labels = read_linear_data()

w = tf.Variable(tf.truncated_normal([5, 1], dtype=tf.float32))
b = tf.Variable(tf.zeros([1], dtype=tf.float32))

predictions = tf.matmul(X, w) + b
cost = tf.reduce_mean(tf.pow(predictions - Y, 2))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in list(range(epochs)):
        sess.run(optimizer, feed_dict={X: train_features, Y: train_labels})
        if i % 100 == 0:
            print("epoch", i, sess.run(cost, feed_dict={X: train_features, Y: train_labels}))

    test_cost = sess.run(cost, feed_dict={X: test_features, Y: test_labels})

    print('------------------')
    print('Test error =', test_cost, '\n')
    print(sess.run(predictions, feed_dict={X: test_features, Y: test_labels}))
