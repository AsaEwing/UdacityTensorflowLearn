from __future__ import print_function

# python3 -m tensorflow.tensorboard --logdir=run1:/tmp/ufLearn5/2 --port=6006

import numpy as np
import tensorflow as tf
import six.moves.cPickle as pickle
from six.moves import range

logs_path2 = "/tmp/ufLearn5/2"

pickle_file = 'notMNIST.pickle'

image_size = 28
num_labels = 10
batch_size = 128
num_steps = 5001


def Wx_plus_b(x, w, b):
    with tf.name_scope('Wx_plus_b'):
        # return tf.nn.relu(tf.matmul(x, w) + b)
        return tf.matmul(x, w) + b


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
        # mTrain_image = tf.reshape(mTrain_in, [-1, image_size, image_size, 1], name="mTrain_image")
        mTrain_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name="mTrain_labels")
        # print('# # # # # #', train_in.size, train_image.shape, train_labels.shape)

    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    with tf.name_scope("Layer1"):
        hidden_stddev = np.sqrt(2.0 / 784)
        with tf.name_scope("weights"):
            # w1 = tf.Variable(tf.truncated_normal([image_size * image_size, 1024], stddev=hidden_stddev))
            w1 = tf.Variable(tf.truncated_normal([image_size * image_size, 1024]))
        with tf.name_scope("biases"):
            b1 = tf.Variable(tf.zeros([1024]))
        '''logits_conv1 = Wx_plus_b(mTrain_in, w1, b1)
        valid_logits1 = Wx_plus_b(tf_valid_dataset, w1, b1)
        test_logits1 = Wx_plus_b(tf_test_dataset, w1, b1)'''
        logits_conv1 = tf.nn.relu(Wx_plus_b(mTrain_in, w1, b1))
        valid_logits1 = tf.nn.relu(Wx_plus_b(tf_valid_dataset, w1, b1))
        test_logits1 = tf.nn.relu(Wx_plus_b(tf_test_dataset, w1, b1))

    with tf.name_scope("Layer2"):
        hidden_stddev = np.sqrt(2.0 / 1024)
        with tf.name_scope("weights"):
            # w2 = tf.Variable(tf.truncated_normal([1024, 500], stddev=hidden_stddev))
            w2 = tf.Variable(tf.truncated_normal([1024, 500]))
        with tf.name_scope("biases"):
            b2 = tf.Variable(tf.zeros([500]))
        # dropout
        with tf.name_scope('dropout'):
            keep_prob2 = tf.placeholder(tf.float32)
            # keep_prob2 = 0.5
            logits_drop = tf.nn.dropout(logits_conv1, keep_prob2)
        '''logits_conv2 = Wx_plus_b(logits_conv1, w2, b2)
        valid_logits2 = Wx_plus_b(valid_logits1, w2, b2)
        test_logits2 = Wx_plus_b(test_logits1, w2, b2)'''
        logits_conv2 = tf.nn.relu(Wx_plus_b(logits_drop, w2, b2))
        valid_logits2 = tf.nn.relu(Wx_plus_b(valid_logits1, w2, b2))
        test_logits2 = tf.nn.relu(Wx_plus_b(test_logits1, w2, b2))

    with tf.name_scope("Layer3"):
        hidden_stddev = np.sqrt(2.0 / 500)
        with tf.name_scope("weights"):
            # w3 = tf.Variable(tf.truncated_normal([500, 800], stddev=hidden_stddev))
            w3 = tf.Variable(tf.truncated_normal([500, 800]))
        with tf.name_scope("biases"):
            b3 = tf.Variable(tf.zeros([800]))
        # dropout
        with tf.name_scope('dropout'):
            keep_prob3 = tf.placeholder(tf.float32)
            # keep_prob3 = 0.5
            logits_drop = tf.nn.dropout(logits_conv2, keep_prob3)
            # valid_drop = tf.nn.dropout(valid_logits3, keep_prob)
            # test_drop = tf.nn.dropout(test_logits3, keep_prob)
        '''logits_conv3 = Wx_plus_b(logits_conv2, w3, b3)
        valid_logits3 = Wx_plus_b(valid_logits2, w3, b3)
        test_logits3 = Wx_plus_b(test_logits2, w3, b3)'''
        logits_conv3 = tf.nn.relu(Wx_plus_b(logits_drop, w3, b3))
        valid_logits3 = tf.nn.relu(Wx_plus_b(valid_logits2, w3, b3))
        test_logits3 = tf.nn.relu(Wx_plus_b(test_logits2, w3, b3))

    # full connection
    '''with tf.name_scope('layer_full'):
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
            y_conv = tf.nn.softmax(h_fc2)'''

    with tf.name_scope("Layer4"):
        with tf.name_scope("weights"):
            hidden_stddev = np.sqrt(2.0 / 800)
            # w4 = tf.Variable(tf.truncated_normal([800, 10], stddev=hidden_stddev))
            w4 = tf.Variable(tf.truncated_normal([800, 10]))
        with tf.name_scope("biases"):
            b4 = tf.Variable(tf.zeros([10]))
        # dropout
        '''with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            logits_drop = tf.nn.dropout(logits_conv3, keep_prob)
            # valid_drop = tf.nn.dropout(valid_logits3, keep_prob)
            # test_drop = tf.nn.dropout(test_logits3, keep_prob)'''
        logits_conv4 = Wx_plus_b(logits_conv3, w4, b4)
        valid_logits4 = Wx_plus_b(valid_logits3, w4, b4)
        test_logits4 = Wx_plus_b(test_logits3, w4, b4)

        with tf.name_scope("softmax"):
            y_conv = tf.nn.softmax(logits_conv4)

            valid_prediction = tf.nn.softmax(valid_logits4)
            test_prediction = tf.nn.softmax(test_logits4)

    with tf.name_scope('cross_entropy'):
        beta = 0.0001
        l2_loss = beta * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4))
        l2_loss += beta * (tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2) + tf.nn.l2_loss(b3) + tf.nn.l2_loss(b4))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=mTrain_labels, logits=logits_conv4)) \
           + l2_loss
        # loss = -tf.reduce_sum(mTrain_labels * tf.log(y_conv))

    # Optimizer.
    with tf.name_scope('train'):
        global_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
        starter_learning_rate = 0.0001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.005, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        # optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

        # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
        # optimizer = tf.train.AdamOptimizer(0.0002).minimize(loss)

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
        offset_range = train_labels.shape[0] - batch_size
        offset = (step * batch_size) % offset_range
        # offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.

        # feed_dict = {mTrain_in: batch_data, mTrain_labels: batch_labels}
        # feed_dict = {mTrain_in: batch_data, mTrain_labels: batch_labels, keep_prob: 1.0}
        feed_dict = {mTrain_in: batch_data, mTrain_labels: batch_labels, keep_prob2: 1.0, keep_prob3: 1.0}
        _, l, predictions = session.run([optimizer, loss, y_conv], feed_dict=feed_dict)

        if step % 200 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("   Minibatch accuracy: %.1f%%" % accuracy2(predictions, batch_labels))
            print("   Validation accuracy: %.1f%%" % accuracy2(valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy2(test_prediction.eval(), test_labels))

writer2.close()
