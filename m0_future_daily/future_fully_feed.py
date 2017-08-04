import urllib.request
import csv
import os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# data source and data sets
DATA_URL = "http://stock2.finance.sina.com.cn/futures/api/json.php/IndexService.getInnerFuturesDailyKLine?symbol=M0"
RAW_DATA_FILE = "raw_data.csv"
TRAIN_FILE = "train_data.csv"
TEST_FILE = "test_data.csv"


def data_process():
    url = urllib.request.urlopen(DATA_URL)
    data = url.read().decode()
    data = data.replace("[[", "")
    data = data.replace("]]", "")
    data = data.replace("\"", "")
    data = data.split("],[")
    for i in range(len(data)):
        data[i] = data[i].split(",")
    print("days: " + str(len(data)))

    index = 0
    for i in range(len(data) - 1):
        data[i][0] = index
        index += 1
        if float(data[i][4]) < float(data[i + 1][4]):
            data[i].append(1)  # closing price will rise tomorrow
            data[i].append(0)
        else:
            data[i].append(0)  # closing price will not rise tomorrow
            data[i].append(1)
    del data[-1]

    with open(RAW_DATA_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in data:
            writer.writerow(line)
        print("wrote data successfully: " + RAW_DATA_FILE)

    with open(TRAIN_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in data[:2926]:
            writer.writerow(line)
        print("wrote data successfully: " + TRAIN_FILE)

    with open(TEST_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in data[2926:]:
            writer.writerow(line)
        print("wrote data successfully: " + TEST_FILE)


def read_data():
    with open(TRAIN_FILE, 'r') as csvfile:
        reader = csv.reader(csvfile)
        train_list = list(reader)
        train_feature = []
        train_output = []
        for item in train_list:
            train_feature.append(item[:6])
            train_output.append(item[6:])

    with open(TEST_FILE, 'r') as csvfile:
        reader = csv.reader(csvfile)
        test_list = list(reader)
        test_feature = []
        test_output = []
        for item in test_list:
            test_feature.append(item[:6])
            test_output.append(item[6:])
    print(train_feature)
    print(train_output)
    print(test_feature)
    print(test_output)
    return train_feature, train_output, test_feature, test_output

# Basic model parameters as external flags.
FLAGS = None

# Parameters
learning_rate = 0.001
training_epochs = 10
batch_size = 50
display_step = 1

# Network Parameters
n_hidden_1 = 10  # 1st layer number of features
n_hidden_2 = 10  # 2nd layer number of features
n_input = 6      # Number of feature
n_classes = 2    # Number of classes to predict

# tf Graph input
x = tf.placeholder("float", [None, n_input])    # train_feature
y = tf.placeholder("float", [None, n_classes])  # train_output

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.matmul(x, weights['h1']) + biases['b1']
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.matmul(layer_1, weights['h2']) + biases['b2']
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=pred, logits=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initializing the variables and Read data
init = tf.initialize_all_variables()
train_feature, train_output, test_feature, test_output = read_data()

train_feature = tf.constant(train_feature)
train_output = tf.constant(train_output)
test_feature = tf.constant(test_feature)
test_output = tf.constant(test_output)

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_loss = 0.
        total_batch = int(len(train_feature)/batch_size)
        X_batches = np.array_split(train_feature, total_batch)
        Y_batches = np.array_split(train_output, total_batch)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # batch_y.shape = (batch_y.shape[0], 1)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_loss += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_loss))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: test_feature, y: test_output}))
    global result
    result = tf.argmax(pred, 1).eval({x: test_feature, y: test_output})
