# __author__ = 'chapter'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# python3 -m tensorflow.tensorboard --logdir=run1:/tmp/mnist_logs3/1 --port=6006

logs_path = "/tmp/mnist_logs3/1"
learnStep = 5000
# learnStep = 20000


def weight_variable(shape):
    with tf.name_scope('weights'):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


def bias_variable(shape):
    with tf.name_scope('biases'):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


def conv2d(x, W):
    with tf.name_scope('conv2d'):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    with tf.name_scope('max_pool'):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def WconvX_plus_b(x, w, b):
    with tf.name_scope('WconvX_plus_b'):
        return tf.nn.relu(conv2d(x, w) + b)


def Wx_plus_b(x, w, b):
    with tf.name_scope('Wx_plus_b'):
        return tf.matmul(x, w) + b


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("Download Done!")

# session = tf.InteractiveSession()

graph = tf.Graph()
with graph.as_default():

    # input
    with tf.name_scope('input'):
        x_in = tf.placeholder(tf.float32, [None, 784], name="x_input")
        x_image = tf.reshape(x_in, [-1, 28, 28, 1], name="x_reshape")
        y_in = tf.placeholder(tf.float32, [None, 10], name="y_input")

    # conv layer-1
    with tf.name_scope('layer1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = WconvX_plus_b(x_image, W_conv1, b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    # conv layer-2
    with tf.name_scope('layer2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = WconvX_plus_b(h_pool1, W_conv2, b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    # full connection
    with tf.name_scope('layer_full'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        with tf.name_scope('reshape_pool'):
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = Wx_plus_b(h_pool2_flat, W_fc1, b_fc1)

        # dropout
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # output layer: softmax
        with tf.name_scope("softmax"):
            W_fc2 = weight_variable([1024, 10])
            b_fc2 = bias_variable([10])
            h_fc2 = Wx_plus_b(h_fc1_drop, W_fc2, b_fc2)
            y_conv = tf.nn.softmax(h_fc2)

    # model training
    with tf.name_scope('cross_entropy'):
        cross_entropy = -tf.reduce_sum(y_in * tf.log(y_conv))
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        # with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_in, 1))
        # with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session(graph=graph) as session:

    session.run(tf.initialize_all_variables())
    writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

    for i in range(learnStep):
        batch = mnist.train.next_batch(50)

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x_in: batch[0], y_in: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x_in: batch[0], y_in: batch[1], keep_prob: 0.5})

    # accuracy on test
    print("test accuracy %g" % (accuracy.eval(feed_dict={x_in: mnist.test.images, y_in: mnist.test.labels, keep_prob: 1.0})))

writer.close()
