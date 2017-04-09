from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import sys
import os
from ClassOfTensorflow2 import Timer
from ClassOfTensorflow2 import Graph


# python3 -m tensorflow.tensorboard --logdir=run1:/tmp/ufLearn4/2 --port=6006


class Constant(object):
    def __init__(self, is1D):
        self.fileName = "test04_2"
        self.size_image = 28
        self.size_batch = 8

        self.num_labels = 10
        self.num_steps = 50001

        self.patch_size = 5
        self.depth = 16
        self.depth2 = 32
        self.num_hidden = 64
        self.num_channels = 1

        if type(is1D) == bool:
            if is1D:
                self.size_input = self.size_image * self.size_image
        else:
            print("'is1D' is not bool")
            sys.exit()

        self._keep_prob = [0.9, 0.7, 0.9, None]
        self.layerCount = 4
        self.layer_input_dim = [[self.patch_size, self.patch_size, self.num_channels, self.depth],
                                [self.patch_size, self.patch_size, self.depth, self.depth2],
                                [self.size_image // 4 * self.size_image // 4 * self.depth2, self.num_hidden],
                                [self.num_hidden, self.num_labels]]

        self.layer_output_dim = [[self.depth],
                                 [self.depth2],
                                 [self.num_hidden],
                                 [self.num_labels]]
        self.layer_isRelu = [True, True, True, False]
        self.layer_kind = ["Conv", "Conv", "Normal", "Normal"]
        self.pool_kind = ["Max", "Max", None, None]

        self.loss_beta_w = 0.0005
        self.loss_beta_b = 0.0001

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


logs_path2 = "/tmp/ufLearn4/2"

pickle_file = 'notMNIST.pickle'
num_stddev = 0.1

constant = Constant(False)
trainTimer = Timer(constant.fileName)


def trainEnd():
    print("\n~ # Train finish")
    print("   Minibatch accuracy max    : %.2f%%" % constant.maxAccuracyMinibatch)
    print("   Validation accuracy max   : %.2f%%" % constant.maxAccuracyValidation)
    print("   Test accuracy max         : %.2f%%" % constant.maxAccuracyTest)

    trainTimer.now()


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
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)

    def reformat(dataset, labels):
        dataset = dataset.reshape((-1, constant.size_image,
                                   constant.size_image,
                                   constant.num_channels)).astype(np.float32)
        labels = (np.arange(constant.num_labels) == labels[:, None]).astype(np.float32)
        return dataset, labels

    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    def accuracy(prediction, labels):
        return 100.0 * np.sum(np.argmax(prediction, 1) == np.argmax(labels, 1)) / prediction.shape[0]

    graph = tf.Graph()

    with graph.as_default():
        # Input data.
        train_image_in = tf.placeholder(
            tf.float32, shape=(constant.size_batch, constant.size_image, constant.size_image, constant.num_channels))
        train_labels_in = tf.placeholder(tf.float32, shape=(constant.size_batch, constant.num_labels))
        valid_image_in = tf.constant(valid_dataset)
        test_image_in = tf.constant(test_dataset)

        self_graph = Graph(constant.fileName)

        trainLayer = self_graph.def_train_Layer(self_graph,
                                                constant.layerCount,
                                                train_image_in,
                                                constant.layer_kind)
        trainLayer.set_LayerVar(constant.layer_isRelu, constant.keep_prob, 0.1)
        trainLayer.set_LayerSize(constant.layer_input_dim, constant.layer_output_dim)
        trainLayer.set_LayerConv(strides=[1, 1, 1, 1], padding="SAME")
        trainLayer.set_LayerPool(kind=constant.pool_kind, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        trainLayer.finish()

        self_graph.softmax()
        self_graph.cross_entropy(constant.loss_beta_w, constant.loss_beta_b, train_labels_in)

        valid_logits_List, valid_prediction = self_graph.test_logits("valid", valid_image_in)
        test_logits_List, test_prediction = self_graph.test_logits("test", test_image_in)

        self_graph.train(True, 0.05, "GradientDescentOptimizer", 0.97, 2000)

    with tf.Session(graph=graph) as session:
        session.run(tf.initialize_all_variables())

        writer = tf.train.SummaryWriter(logs_path2, graph=tf.get_default_graph())

        trainTimer.now("Initialize")

        print('Initialized')
        for step in range(constant.num_steps):
            offset = (step * constant.size_batch) % (train_labels.shape[0] - constant.size_batch)
            batch_data = train_dataset[offset:(offset + constant.size_batch), :, :, :]
            batch_labels = train_labels[offset:(offset + constant.size_batch), :]
            feed_dict = {train_image_in: batch_data, train_labels_in: batch_labels}
            _, l, predictions = session.run(
                [self_graph.optimizer, self_graph.loss, self_graph.softmax_out], feed_dict=feed_dict)
            '''if step % 50 == 0:
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(
                    valid_prediction.eval(), valid_labels))'''

            if step % 100 == 0:
                accuracyMinibatch = accuracy(predictions, batch_labels)
                accuracyValidation = accuracy(valid_prediction.eval(), valid_labels)
                accuracyTest = accuracy(test_prediction.eval(), test_labels)

                constant.countPrintStep += 1

                def accuracyAll():
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

                accuracyAll()

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

            if step % 5000 == 0 and step != 0:
                trainEnd()
                print("######################################")
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

    writer.close()
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
