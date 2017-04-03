from __future__ import print_function

# python3 -m tensorflow.tensorboard --logdir=run1:/tmp/ufLearn2_main/1 --port=6006

import numpy as np
import tensorflow as tf
import six.moves.cPickle as pickle
from six.moves import range
import time
import sys
import os


class Timer(object):
    def __init__(self, name):
        self.Time_AllStart = time.time() * 1000
        self.Time_End = 0
        self.name = name

        print("\n~ # %s 程式開始\n" % self.name)
        return

    def operation_time(self, deltaTime):
        mHour = 0
        mMin = 0
        mS = 0
        mMs = 0
        if deltaTime > 3600000:
            mHour = int(deltaTime / 3600000)
            deltaTime = deltaTime % 3600000
        if deltaTime > 60000:
            mMin = int(deltaTime / 60000)
            deltaTime = deltaTime % 60000
        if deltaTime > 1000:
            mS = int(deltaTime / 1000)
            mMs = deltaTime % 1000

        return [mHour, mMin, mS, mMs]

    def now(self, remind=""):
        self.Time_End = time.time() * 1000
        deltaTime = float(self.Time_End - self.Time_AllStart)
        timeList = self.operation_time(deltaTime)

        if timeList[0] > 0:
            print('\n~ # %s 已過時間：%d h, %d min, %d s, %d ms' % (
                remind, timeList[0], timeList[1], timeList[2], timeList[3]))

        elif timeList[1] > 0:
            print('\n~ # %s 已過時間：%d h, %d min, %d s, %d ms' % (
                remind, timeList[0], timeList[1], timeList[2], timeList[3]))
        elif timeList[2] > 0:
            print('\n~ # %s 已過時間：%d h, %d min, %d s, %d ms' % (
                remind, timeList[0], timeList[1], timeList[2], timeList[3]))
        else:
            print('\n~ # %s 已過時間：%d h, %d min, %d s, %d ms' % (
                remind, timeList[0], timeList[1], timeList[2], timeList[3]))

        return

    def end(self):
        self.Time_End = time.time() * 1000
        deltaTime = float(self.Time_End - self.Time_AllStart)
        timeList = self.operation_time(deltaTime)

        if timeList[0] > 0:
            print('\n~ # %s 程式結束，時間共：%d h, %d min, %d s, %d ms' % (
                self.name, timeList[0], timeList[1], timeList[2], timeList[3]))

        elif timeList[1] > 0:
            print('\n~ # %s 程式結束，時間共：%d h, %d min, %d s, %d ms' % (
                self.name, timeList[0], timeList[1], timeList[2], timeList[3]))
        elif timeList[2] > 0:
            print('\n~ # %s 程式結束，時間共：%d h, %d min, %d s, %d ms' % (
                self.name, timeList[0], timeList[1], timeList[2], timeList[3]))
        else:
            print('\n~ # %s 程式結束，時間共：%d h, %d min, %d s, %d ms' % (
                self.name, timeList[0], timeList[1], timeList[2], timeList[3]))

        return


class Constant(object):
    def __init__(self, is1D):
        self.size_image = 28
        self.size_batch = 128

        self.num_labels = 10
        self.num_steps = 80001

        self._keep_prob = [1.0, 0.7, 0.5, None]

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

        if type(is1D) == bool:
            if is1D:
                self.size_input = self.size_image * self.size_image
        else:
            print("'is1D' is not bool")
            sys.exit()

        return

    def keep_prob(self, index):
        return self._keep_prob[index - 1]


trainTimer = Timer("test02_main")

logs_path2 = "/tmp/ufLearn2_main/1"

pickle_file = 'notMNIST.pickle'
num_stddev = 0.1

constant = Constant(True)

'''
Minibatch loss at step 80000: 0.307779
   Minibatch accuracy    : 100.00%
   Validation accuracy   : 90.15%
   Test accuracy         : 96.02%

~ # Train finish
   Minibatch accuracy max    : 100.00%
   Validation accuracy max   : 90.77%
   Test accuracy max         : 96.31%

~ # test02_main 程式結束，時間共：0020.134 min
'''
'''
Minibatch loss at step 80000: 0.297019
   Minibatch accuracy    : 100.00%
   Validation accuracy   : 90.56%
   Test accuracy         : 96.08%

~ # Train finish
   Minibatch accuracy max    : 100.00%
   Validation accuracy max   : 90.87%
   Test accuracy max         : 96.28%

~ # test02_main 程式結束，時間共：0024.018 min
'''
'''
Minibatch loss at step 80000: 0.304564
   Minibatch accuracy    : 100.00%
   Validation accuracy   : 90.84%
   Test accuracy         : 96.03%

~ #  已過時間：0 h, 21 min, 19 s, 102 ms

~ # Train finish
   Minibatch accuracy max    : 100.00%
   Validation accuracy max   : 90.84%
   Test accuracy max         : 96.14%

~ # test02_main 程式結束，時間共：0 h, 21 min, 19 s, 102 ms
'''
'''
Minibatch loss at step 80000: 0.235925
   Minibatch accuracy    : 100.00%
   Validation accuracy   : 89.81%
   Test accuracy         : 95.63%

~ #  已過時間：0 h, 22 min, 46 s, 955 ms

~ # Train finish
   Minibatch accuracy max    : 100.00%
   Validation accuracy max   : 90.20%
   Test accuracy max         : 95.78%

~ # test02_main 程式結束，時間共：0 h, 22 min, 46 s, 957 ms
'''

'''
Minibatch loss at step 80000 , 401 : 0.247177
   Minibatch accuracy    : 100.00% , -339, 75
   Validation accuracy   : 89.93% , -193, 61
   Test accuracy         : 95.67% , -189, 47

~ #  已過時間：0 h, 21 min, 49 s, 873 ms

~ # Train finish
   Minibatch accuracy max    : 100.00%
   Validation accuracy max   : 90.12%
   Test accuracy max         : 95.79%

~ # test02_main 程式結束，時間共：0 h, 21 min, 49 s, 873 ms
'''
'''
Minibatch loss at step 80000 , 401 : 0.249254
   Minibatch accuracy    : 100.00% , -355, 75
   Validation accuracy   : 89.97% , -201, 67
   Test accuracy         : 95.57% , -181, 39

~ #  已過時間：0 h, 34 min, 8 s, 542 ms

~ # Train finish
   Minibatch accuracy max    : 100.00%
   Validation accuracy max   : 90.12%
   Test accuracy max         : 95.82%

~ # test02_main 程式結束，時間共：0 h, 34 min, 8 s, 542 ms
'''
'''
Minibatch loss at step 80000 , 401 : 0.264419
   Minibatch accuracy    : 100.00% , -355, 75
   Validation accuracy   : 90.07% , -197, 67
   Test accuracy         : 95.73% , -213, 45

~ #  已過時間：0 h, 34 min, 22 s, 48 ms

~ # Train finish
   Minibatch accuracy max    : 100.00%
   Validation accuracy max   : 90.50%
   Test accuracy max         : 96.04%

~ # test02_main 程式結束，時間共：0 h, 34 min, 22 s, 48 ms
'''
'''
Minibatch loss at step 80000 , 401 : 0.098670 ,rate 7.81464e-05 :
   Minibatch accuracy    : 100.00% , -331, 65
   Validation accuracy   : 90.80% , -247, 27
   Test accuracy         : 96.26% , -263, 29

~ #  已過時間：0 h, 20 min, 35 s, 531 ms

~ # Train finish
   Minibatch accuracy max    : 100.00%
   Validation accuracy max   : 90.88%
   Test accuracy max         : 96.39%

~ # Train finish
   Minibatch accuracy max    : 100.00%
   Validation accuracy max   : 90.88%
   Test accuracy max         : 96.39%

~ # test02_main 程式結束，時間共：0 h, 20 min, 35 s, 532 ms
'''

def w_var(shape):
    with tf.name_scope('W'):
        initial = tf.truncated_normal(shape, stddev=num_stddev)
        return tf.Variable(initial)


def b_var(shape):
    with tf.name_scope('B'):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


def conv_2d(x, W):
    with tf.name_scope('conv2d'):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    with tf.name_scope('maxPool'):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def W_conv_X_plus_b(inX, w, b, isRelu):
    if type(isRelu) == bool:
        # print("'isNeedRelu' is bool")

        if isRelu:
            with tf.name_scope('WcX_b_Relu'):
                return tf.nn.relu(conv_2d(inX, w) + b)
        elif not isRelu:
            with tf.name_scope('WcX_b'):
                return conv_2d(inX, w) + b
    else:
        print("'isNeedRelu' is not bool")
        sys.exit()


def Wx_plus_b(inX, w, b, isRelu):
    if type(isRelu) == bool:
        # print("'isNeedRelu' is bool")

        if isRelu:
            with tf.name_scope('WX_b_Relu'):
                return tf.nn.relu(tf.matmul(inX, w) + b)
        elif not isRelu:
            with tf.name_scope('WX_b'):
                return tf.matmul(inX, w) + b
    else:
        print("'isNeedRelu' is not bool")
        sys.exit()


'''def Layer(layer_num, input_tensor, input_dim, output_dim, isNeedRelu, isNeedConv, keep_prob):
    def w_var(shape):
        with tf.name_scope('W'):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

    def b_var(shape):
        with tf.name_scope('B'):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

    def conv_2d(x, W):
        with tf.name_scope('conv2d'):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        with tf.name_scope('maxPool'):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def W_conv_X_plus_b(inX, w, b, isRelu):
        if type(isRelu) == bool:
            # print("'isNeedRelu' is bool")

            if isRelu:
                with tf.name_scope('WcX_b_Relu'):
                    return tf.nn.relu(conv_2d(inX, w) + b)
            elif not isRelu:
                with tf.name_scope('WcX_b'):
                    return conv_2d(inX, w) + b
        else:
            print("'isNeedRelu' is not bool")
            return None

    def Wx_plus_b(inX, w, b, isRelu):
        if type(isRelu) == bool:
            # print("'isNeedRelu' is bool")

            if isRelu:
                with tf.name_scope('WX_b_Relu'):
                    return tf.nn.relu(tf.matmul(inX, w) + b)
            elif not isRelu:
                with tf.name_scope('WX_b'):
                    return tf.matmul(inX, w) + b
        else:
            print("'isNeedRelu' is not bool")
            return None

    def nn_layer(layerNum, inputT, inputS, outputS, isRelu, isConv, mKeep_prob):
        layerName = "Layer%d" % layerNum
        print(layerName)

        if type(isRelu) == bool and type(isConv) == bool:
            with tf.name_scope(layerName):
                w = w_var([inputS, outputS])
                b = b_var([outputS])

                if mKeep_prob is not None:

                    if isNeedConv:
                        logits = W_conv_X_plus_b(inputT, w, b, isRelu)
                    else:
                        logits = Wx_plus_b(inputT, w, b, isRelu)
                    with tf.name_scope('dropout'):
                        logits_final = tf.nn.dropout(logits, mKeep_prob)
                else:
                    if isNeedConv:
                        logits_final = W_conv_X_plus_b(inputT, w, b, isRelu)
                    else:
                        logits_final = Wx_plus_b(inputT, w, b, isRelu)

            return logits_final
        else:
            return None

    return nn_layer(layer_num, input_tensor, input_dim, output_dim, isNeedRelu, isNeedConv, keep_prob)'''


class Layer(object):
    def __init__(self, layer_num, input_tensor, input_dim, output_dim, isNeedRelu, isNeedConv, keep_prob):
        if type(layer_num) == int:
            if layer_num > 0:
                self.layerName = "Layer%02d" % layer_num
            else:
                print("'layer_num' need > 0")
                sys.exit()
        else:
            print("'layer_num' is not int")
            sys.exit()

        self.w = None
        self.b = None
        self.layer = None

        if type(isNeedRelu) == bool and type(isNeedConv) == bool:
            if keep_prob is None \
                    or type(keep_prob) == float \
                    or type(keep_prob) == int:

                with tf.name_scope(self.layerName):
                    self.w = self.w_var([input_dim, output_dim])
                    self.b = self.b_var([output_dim])
                    self.layer = self.nn_layer(layer_num, input_tensor,
                                               input_dim, output_dim,
                                               isNeedRelu, isNeedConv,
                                               keep_prob)
            else:
                print("'keep_prob' is not None or int or float")
                sys.exit()
        else:
            if type(isNeedRelu) != bool:
                print("'isNeedRelu' is not bool")
            if type(isNeedConv) != bool:
                print("'isNeedConv' is not bool")
            sys.exit()
        return

    def w_var(self, shape):
        with tf.name_scope('W'):
            initial = tf.truncated_normal(shape, stddev=num_stddev)
            return tf.Variable(initial)

    def b_var(self, shape):
        with tf.name_scope('B'):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

    def conv_2d(self, x, W):
        with tf.name_scope('conv2d'):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        with tf.name_scope('maxPool'):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def W_conv_X_plus_b(self, inX, w, b, isRelu):
        if isRelu:
            with tf.name_scope('WcX_b_Relu'):
                return tf.nn.relu(self.conv_2d(inX, w) + b)
        elif not isRelu:
            with tf.name_scope('WcX_b'):
                return self.conv_2d(inX, w) + b

    def Wx_plus_b(self, inX, w, b, isRelu):
        if isRelu:
            with tf.name_scope('WX_b_Relu'):
                return tf.nn.relu(tf.matmul(inX, w) + b)
        elif not isRelu:
            with tf.name_scope('WX_b'):
                return tf.matmul(inX, w) + b

    def nn_layer(self, layerNum, inputT, inputS, outputS, isRelu, isConv, mKeep_prob):

        if mKeep_prob is not None:

            if isConv:
                logits = self.W_conv_X_plus_b(inputT, self.w, self.b, isRelu)
            else:
                logits = self.Wx_plus_b(inputT, self.w, self.b, isRelu)
            with tf.name_scope('dropout'):
                logits_final = tf.nn.dropout(logits, mKeep_prob)
        else:
            if isConv:
                logits_final = self.W_conv_X_plus_b(inputT, self.w, self.b, isRelu)
            else:
                logits_final = self.Wx_plus_b(inputT, self.w, self.b, isRelu)

        return logits_final


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
        # keep_prob = []

        with tf.name_scope('input'):
            mTrain_in = tf.placeholder(tf.float32,
                                       shape=(constant.size_batch, constant.size_input),
                                       name="mTrain_in")
            # mTrain_image = tf.reshape(mTrain_in, [-1, image_size, image_size, 1], name="mTrain_image")
            mTrain_labels = tf.placeholder(tf.float32,
                                           shape=(constant.size_batch, constant.num_labels),
                                           name="mTrain_labels")

        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # hidden_stddev = np.sqrt(2.0 / 784)
        # 1500
        Layer1 = Layer(1, mTrain_in, constant.size_input, 1024, True, False, constant.keep_prob(1))

        # hidden_stddev = np.sqrt(2.0 / 1024)
        # 800
        Layer2 = Layer(2, Layer1.layer, 1024, 256, True, False, constant.keep_prob(2))

        # hidden_stddev = np.sqrt(2.0 / 500)
        # 300
        Layer3 = Layer(3, Layer2.layer, 256, 64, True, False, constant.keep_prob(3))

        # hidden_stddev = np.sqrt(2.0 / 800)
        # 10
        Layer4 = Layer(4, Layer3.layer, 64, constant.num_labels, False, False, constant.keep_prob(4))

        with tf.name_scope("valid"):
            valid_logits1 = Wx_plus_b(tf_valid_dataset, Layer1.w, Layer1.b, True)
            valid_logits2 = Wx_plus_b(valid_logits1, Layer2.w, Layer2.b, True)
            valid_logits3 = Wx_plus_b(valid_logits2, Layer3.w, Layer3.b, True)
            valid_logits4 = Wx_plus_b(valid_logits3, Layer4.w, Layer4.b, False)
            valid_prediction = tf.nn.softmax(valid_logits4)

        with tf.name_scope("test"):
            test_logits1 = Wx_plus_b(tf_test_dataset, Layer1.w, Layer1.b, True)
            test_logits2 = Wx_plus_b(test_logits1, Layer2.w, Layer2.b, True)
            test_logits3 = Wx_plus_b(test_logits2, Layer3.w, Layer3.b, True)
            test_logits4 = Wx_plus_b(test_logits3, Layer4.w, Layer4.b, False)
            test_prediction = tf.nn.softmax(test_logits4)

        with tf.name_scope("softmax"):
            y_conv = tf.nn.softmax(Layer4.layer)

        with tf.name_scope('cross_entropy'):
            l2_loss_w = tf.nn.l2_loss(Layer1.w)
            l2_loss_w += tf.nn.l2_loss(Layer2.w)
            l2_loss_w += tf.nn.l2_loss(Layer3.w)
            l2_loss_w += tf.nn.l2_loss(Layer4.w)
            l2_loss_w = constant.loss_beta_w * l2_loss_w

            l2_loss_b = tf.nn.l2_loss(Layer1.b)
            l2_loss_b += tf.nn.l2_loss(Layer2.b)
            l2_loss_b += tf.nn.l2_loss(Layer3.b)
            l2_loss_b += tf.nn.l2_loss(Layer4.b)
            l2_loss_b = constant.loss_beta_b * l2_loss_b

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=mTrain_labels,
                                                                          logits=Layer4.layer))
            loss += l2_loss_w + l2_loss_b
            # loss = -tf.reduce_sum(mTrain_labels * tf.log(y_conv))

        # Optimizer.
        with tf.name_scope('train'):
            global_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
            starter_learning_rate = 0.0004
            learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                       global_step, 2000,
                                                       0.97,
                                                       staircase=True)
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
            # optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
            # optimizer = tf.train.AdamOptimizer(0.0002).minimize(loss)

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
            _, l, predictions = session.run([optimizer, loss, y_conv], feed_dict=feed_dict)

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
                                                                            learning_rate.eval()))
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
