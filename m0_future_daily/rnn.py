import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn
from m0_future_daily.data_process import read_data, next_batch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def process_one_hot(label_set):
    new_set = []
    for item in label_set:
        if item == 1:
            new_set.append([0, 1])
        else:
            new_set.append([1, 0])
    return new_set

# import data
train_features, train_labels, test_features, test_labels = read_data()
train_labels = process_one_hot(train_labels)
test_labels = process_one_hot(test_labels)

# Parameters
learning_rate = 1e-6
training_iters = 3000000
batch_size = 2000
display_step = 10

# Network Parameters
n_input = 5
n_steps = 1
n_hidden = 1000
n_classes = 2

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # lstm_cell = rnn.GRUCell(n_hidden)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

acc_list = []
loss_list = []
iter_list = []
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = next_batch(batch_size, train_features, train_labels)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            iter_list.append(step)
            loss_list.append(loss)
            acc_list.append(acc)
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_data = test_features.reshape((-1, n_steps, n_input))
    test_label = test_labels
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

# plot
plt.figure(figsize=(8, 8))
plt.plot(iter_list, loss_list, color='red', linewidth=2)
plt.plot(iter_list, acc_list, color='blue', linewidth=2)
plt.xlabel("steps")
plt.ylabel("loss/acc")
plt.show()