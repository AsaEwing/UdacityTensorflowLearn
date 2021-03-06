from __future__ import print_function

# python3 -m tensorflow.tensorboard --logdir=run1:/tmp/ufLearn2_main/1 --port=6006

import numpy as np
import tensorflow as tf
import six.moves.cPickle as pickle
from six.moves import range
import sys
import os
from ClassOfTensorflow2 import Timer
from ClassOfTensorflow2 import Graph


class Constant(object):
    def __init__(self, is1D):
        self.size_image = 28
        self.size_batch = 128

        self.num_labels = 10
        self.num_steps = 80001

        if type(is1D) == bool:
            if is1D:
                self.size_input = self.size_image * self.size_image
        else:
            print("'is1D' is not bool")
            sys.exit()

        self._keep_prob = [1.0, 0.7, 0.5, None]
        self.layerCount = 4
        self.layer_input_dim = [[self.size_input, 1024],
                                [1024, 256],
                                [256, 64],
                                [64, 10]]
        self.layer_output_dim = [[1024], [256], [64], [self.num_labels]]
        self.layer_isRelu = [True, True, True, False]
        self.layer_kind = ["Normal", "Normal", "Normal", "Normal"]

        self.loss_beta_w = 0.0002
        self.loss_beta_b = 0.00005

        self.maxAccuracyMinibatch = 0
        self.maxAccuracyValidation = 0
        self.maxAccuracyTest = 0

        self.lastAccuracyMinibatch = 0
        self.lastAccuracyValidation = 0
        self.lastAccuracyTest = 0

        self.countAll_accuracyMinibatch = 0
        self.countAll_accuracyValidation = 0
        self.countAll_accuracyTest = 0

        self.countLast_accuracyMinibatch = 0
        self.countLast_accuracyValidation = 0
        self.countLast_accuracyTest = 0

        self.countPrintStep = 0

        return

    def keep_prob(self, index):
        return self._keep_prob[index]


trainTimer = Timer("test02_main")

logs_path2 = "/tmp/ufLearn2_main/1"

pickle_file = 'notMNIST.pickle'
num_stddev = 0.1

constant = Constant(True)


def trainEnd():
    print("\n~ # Train finish")
    print("   Minibatch accuracy max    : %.2f%%" % constant.maxAccuracyMinibatch)
    print("   Validation accuracy max   : %.2f%%" % constant.maxAccuracyValidation)
    print("   Test accuracy max         : %.2f%%" % constant.maxAccuracyTest)


def train():
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        print('pickle_file Training set     :%s , %s' % (train_dataset.shape, train_labels.shape))
        print('pickle_file Validation set   :%s , %s' % (valid_dataset.shape, valid_labels.shape))
        print('pickle_file Test set         :%s , %s' % (test_dataset.shape, test_labels.shape))

    def reformat(dataset, labels):
        dataset = dataset.reshape((-1, constant.size_input)).astype(np.float32)
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels = (np.arange(constant.num_labels) == labels[:, None]).astype(np.float32)
        return dataset, labels

    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    graph = tf.Graph()
    with graph.as_default():

        with tf.name_scope('input'):
            mTrain_in = tf.placeholder(tf.float32,
                                       shape=(constant.size_batch, constant.size_input),
                                       name="mTrain_in")
            mTrain_labels = tf.placeholder(tf.float32,
                                           shape=(constant.size_batch, constant.num_labels),
                                           name="mTrain_labels")

        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        self_graph = Graph("test02")

        trainLayer = self_graph.def_train_Layer(self_graph, constant.layerCount, mTrain_in, constant.layer_kind)
        trainLayer.set_LayerVar(constant.layer_isRelu, constant.keep_prob, 0.1)
        trainLayer.set_LayerSize(constant.layer_input_dim, constant.layer_output_dim)
        trainLayer.finish()

        valid_logits_List, valid_prediction = self_graph.test_logits("valid", tf_valid_dataset)
        test_logits_List, test_prediction = self_graph.test_logits("test", tf_test_dataset)

        self_graph.softmax()
        self_graph.cross_entropy(constant.loss_beta_w, constant.loss_beta_b, mTrain_labels)

        self_graph.train(True, 0.0004, "AdamOptimizer", 0.97, 2000)

        trainTimer.now("Graph OK")

    def accuracy2(prediction, labels):
        with tf.name_scope('Accuracy'):
            return 100.0 * np.sum(np.argmax(prediction, 1) == np.argmax(labels, 1)) / prediction.shape[0]

    # summary_op = tf.merge_all_summaries()

    with tf.Session(graph=graph) as session:
        session.run(tf.initialize_all_variables())

        writer2 = tf.train.SummaryWriter(logs_path2, graph=tf.get_default_graph())

        trainTimer.now("Initialize")

        for step in range(constant.num_steps):

            offset_range = train_labels.shape[0] - constant.size_batch
            offset = (step * constant.size_batch) % offset_range
            # offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + constant.size_batch), :]
            batch_labels = train_labels[offset:(offset + constant.size_batch), :]

            feed_dict = {mTrain_in: batch_data, mTrain_labels: batch_labels}
            # feed_dict = {mTrain_in: batch_data, mTrain_labels: batch_labels, keep_prob: 1.0}
            # feed_dict = {mTrain_in: batch_data, mTrain_labels: batch_labels, keep_prob2: 1.0, keep_prob3: 1.0}
            _, l, predictions = session.run([self_graph.optimizer, self_graph.loss, self_graph.softmax_out],
                                            feed_dict=feed_dict)

            if step % 200 == 0:
                accuracyMinibatch = accuracy2(predictions, batch_labels)
                accuracyValidation = accuracy2(valid_prediction.eval(), valid_labels)
                accuracyTest = accuracy2(test_prediction.eval(), test_labels)

                constant.countPrintStep += 1

                # Max
                if accuracyMinibatch >= constant.maxAccuracyMinibatch:
                    constant.maxAccuracyMinibatch = accuracyMinibatch
                    constant.countAll_accuracyMinibatch += 1
                else:
                    constant.countAll_accuracyMinibatch -= 1

                if accuracyValidation >= constant.maxAccuracyValidation:
                    constant.maxAccuracyValidation = accuracyValidation
                    constant.countAll_accuracyValidation += 1
                else:
                    constant.countAll_accuracyValidation -= 1

                if accuracyTest >= constant.maxAccuracyTest:
                    constant.maxAccuracyTest = accuracyTest
                    constant.countAll_accuracyTest += 1
                else:
                    constant.countAll_accuracyTest -= 1

                # Last
                if accuracyMinibatch >= constant.lastAccuracyMinibatch:
                    constant.countLast_accuracyMinibatch += 1
                else:
                    constant.countLast_accuracyMinibatch -= 1

                if accuracyValidation >= constant.lastAccuracyValidation:
                    constant.countLast_accuracyValidation += 1
                else:
                    constant.countLast_accuracyValidation -= 1

                if accuracyTest >= constant.lastAccuracyTest:
                    constant.countLast_accuracyTest += 1
                else:
                    constant.countLast_accuracyTest -= 1

                constant.lastAccuracyMinibatch = accuracyMinibatch
                constant.lastAccuracyValidation = accuracyValidation
                constant.lastAccuracyTest = accuracyTest

                print("\nMinibatch loss at step %d , %d : %f ,rate %s :" % (step, constant.countPrintStep, l,
                                                                            self_graph.learning_rate.eval()))
                print("   Minibatch accuracy    : %.2f%% , %d, %d" % (accuracyMinibatch,
                                                                      constant.countAll_accuracyMinibatch,
                                                                      constant.countLast_accuracyMinibatch))
                print("   Validation accuracy   : %.2f%% , %d, %d" % (accuracyValidation,
                                                                      constant.countAll_accuracyValidation,
                                                                      constant.countLast_accuracyValidation))
                print("   Test accuracy         : %.2f%% , %d, %d" % (accuracyTest,
                                                                      constant.countAll_accuracyTest,
                                                                      constant.countLast_accuracyTest))

            if step % 5000 == 0:
                trainTimer.now()
                trainEnd()
                print("######################################")

        trainEnd()

    writer2.close()

    trainTimer.end()


if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        print('\nInterrupted\n')
        trainEnd()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
