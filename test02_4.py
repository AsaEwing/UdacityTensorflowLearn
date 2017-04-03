from __future__ import print_function

# python3 -m tensorflow.tensorboard --logdir=run1:/tmp/ufLearn3/2 --port=6006

import numpy as np
import tensorflow as tf
import six.moves.cPickle as pickle
from six.moves import range

logs_path2 = "/tmp/ufLearn3/2"

pickle_file = 'notMNIST.pickle'

image_size = 28
num_labels = 10
batch_size = 128
num_steps = 3001


def conv2d(x, W):
    with tf.name_scope('conv2d'):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        # return tf.nn.conv1d(x, W, stride=1, padding='SAME')


def max_pool_2x2(x):
    with tf.name_scope('max_pool'):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def WconvX_plus_b(x, w, b):
    with tf.name_scope('WconvX_plus_b'):
        return tf.nn.relu(conv2d(x, w) + b)


def Wx_plus_b(x, w, b):
    with tf.name_scope('Wx_plus_b'):
        return tf.nn.relu(tf.matmul(x, w) + b)


with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('pickle_file Training set', train_dataset.shape, train_labels.shape)
    print('pickle_file Validation set', valid_dataset.shape, valid_labels.shape)
    print('pickle_file Test set', test_dataset.shape, test_labels.shape)


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    with tf.name_scope('input'):
        mTrain_in = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size), name="mTrain_in")
        mTrain_image = tf.reshape(mTrain_in, [-1, image_size, image_size, 1], name="mTrain_image")
        mTrain_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name="mTrain_labels")
        # print('# # # # # #', train_in.size, train_image.shape, train_labels.shape)

    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    with tf.name_scope("Layer1"):
        with tf.name_scope("weights"):
            w1 = tf.Variable(tf.truncated_normal([5, 5, 1, num_labels]))
            # w1 = tf.Variable(tf.truncated_normal([784, num_labels]))
        with tf.name_scope("biases"):
            b1 = tf.Variable(tf.zeros([num_labels]))
            # b1 = tf.Variable(tf.zeros([num_labels]))
        with tf.name_scope("Wx_b"):
            # logits = Wx_plus_b(mTrain_in, weights, biases)
            logits_conv1 = WconvX_plus_b(mTrain_image, w1, b1)
        with tf.name_scope("MaxPool"):
            logits_pool1 = max_pool_2x2(logits_conv1)

    with tf.name_scope("Layer2"):
        with tf.name_scope("weights"):
            # num_labels * 2
            # weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
            w2 = tf.Variable(tf.truncated_normal([5, 5, num_labels, 1024]))
        with tf.name_scope("biases"):
            # biases = tf.Variable(tf.zeros([num_labels]))
            b2 = tf.Variable(tf.zeros([1024]))
        with tf.name_scope("Wx_b"):
            # logits = Wx_plus_b(mTrain_in, weights, biases)
            logits_conv2 = WconvX_plus_b(logits_pool1, w2, b2)
        with tf.name_scope("MaxPool"):
            logits_pool2 = max_pool_2x2(logits_conv2)

    # full connection
    with tf.name_scope('layer_full'):
        layerFullSize = int(image_size * image_size / 16 * 1024)
        # layerFullSize = 7 * 7 * num_labels * 2
        w3 = tf.Variable(tf.truncated_normal([layerFullSize, 10]))
        b3 = tf.Variable(tf.zeros([10]))

        with tf.name_scope('reshape_pool'):
            logits_pool2_flat = tf.reshape(logits_pool2, [-1, layerFullSize])
            logits_3 = Wx_plus_b(logits_pool2_flat, w3, b3)
        '''
        # dropout
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            logits_3_drop = tf.nn.dropout(logits_3, keep_prob)'''

        # output layer: softmax
        with tf.name_scope("softmax"):
            # wf = tf.Variable(tf.truncated_normal([1024, 10]))
            # bf = tf.Variable(tf.zeros([10]))
            # logits_f = Wx_plus_b(logits_3_drop, wf, bf)
            y_conv = tf.nn.softmax(logits_3)

    # Predictions for the training, validation, and test data.
    # with tf.name_scope("softmax"):
    #     train_prediction = tf.nn.softmax(y_conv)
    # valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, w1) + b1)
    # test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, w1) + b1)

    with tf.name_scope('cross_entropy'):
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=mTrain_labels, logits=logits))
        loss = -tf.reduce_sum(mTrain_labels * tf.log(y_conv))

    # Optimizer.
    with tf.name_scope('train'):
        # optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

print("### End 6")


def accuracy2(prediction, labels):
    with tf.name_scope('Accuracy'):
        return 100.0 * np.sum(np.argmax(prediction, 1) == np.argmax(labels, 1)) / prediction.shape[0]


# summary_op = tf.merge_all_summaries()


with tf.Session(graph=graph) as session:
    # tf.initialize_all_variables().run()
    session.run(tf.initialize_all_variables())

    writer2 = tf.train.SummaryWriter(logs_path2, graph=tf.get_default_graph())

    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.

        # feed_dict = {mTrain_in: batch_data, mTrain_labels: batch_labels, keep_prob: 1.0}
        feed_dict = {mTrain_in: batch_data, mTrain_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, y_conv], feed_dict=feed_dict)

        if step % 10 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("   Minibatch accuracy: %.1f%%" % accuracy2(predictions, batch_labels))
            # print("   Validation accuracy: %.1f%%" % accuracy2(valid_prediction.eval(), valid_labels))
    # print("Test accuracy: %.1f%%" % accuracy2(test_prediction.eval(), test_labels))

writer2.close()
