import numpy as np
import tensorflow as tf
import time
import sys

num_stddev = 0.09


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
        self.size_batch = 64

        self.num_labels = 10
        self.num_steps = 70001

        self._keep_prob = [None, 0.7, 0.9, 0.9]

        self.loss_beta_w = 0.0002
        self.loss_beta_b = 0.00005

        if type(is1D) == bool:
            if is1D:
                self.size_input = self.size_image * self.size_image
        else:
            print("'is1D' is not bool")
            sys.exit()

        return

    def keep_prob(self, index):
        return self._keep_prob[index - 1]


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
    def __init__(self, layer_num, input_tensor, input_dim, output_dim, isNeedRelu, isNeedConv, keep_prob=None):
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
