import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# build the computational graph
# add two constant() ops to default graph
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])

# add one matmul() op to default graph
product = tf.matmul(matrix1, matrix2)

# run the computational graph
# launch default graph
sess = tf.Session()

# use "product" as the parameter of run(), which means we want the output of product
# result is an object of 'ndarray' from numpy
result1 = sess.run(product)
print("result1: " + str(result1))

# close the session
sess.close()

# use with to omit close()
with tf.Session() as sess:
    result2 = sess.run(product)
    print("result2: " + str(result2))

# Variables
# create a variable
state = tf.Variable(0, name="counter")

# create an op to increase state
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# have to init variable first
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

# Fetch
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(intermed, input1)

with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print(result)

# Feed
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    result = sess.run(output, feed_dict={input1: 7, input2: 2})
    print(result)

# Constant
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)  # also tf.float32 implicitly
print(node1, node2)

# Run a session
sess = tf.Session()
print(sess.run([node1, node2]))

# add op
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))

# Placeholder(Feed)
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3.  # 3. provides a shortcut for tf.constant(3.0)
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# Variable
W = tf.Variable([0.3])
b = tf.Variable([-0.3])
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

y = tf.placeholder(tf.float32)  # desired values
squared_deltas = tf.square(linear_model - y)  # loss function
loss = tf.reduce_sum(squared_deltas)  # loss function
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)  # reset values to incorrect defaults.
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))
