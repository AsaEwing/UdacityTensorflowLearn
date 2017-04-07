import numpy as np
import tensorflow as tf
import time
import sys


class Timer(object):
    def __init__(self, name):
        self.Time_AllStart = time.time() * 1000
        self.Time_End = 0
        self.name = name

        print("\n~ # %s 程式開始\n" % self.name)
        return

    @staticmethod
    def operation_time(deltaTime):
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


class LayerNormal(object):
    def __init__(self, name, inputTensor, isRelu, keep_prob=None):
        self.name = name
        self.input = inputTensor
        self.isRelu = isRelu
        self.keep_prob = keep_prob

        self.w = None
        self.b = None
        self.layer = None
        self.w_constant = None
        self.b_constant = None
        return

    def _w_init(self):
        with tf.name_scope('W'):
            if self.w_constant['stddev'] is not None:
                self.w = tf.truncated_normal(shape=self.w_constant['shape'],
                                             stddev=self.w_constant['stddev'])
            else:
                self.w = tf.truncated_normal(shape=self.w_constant['shape'])
            self.w = tf.Variable(self.w)

    def _b_init(self):
        with tf.name_scope('B'):
            self.b = tf.constant(0.1, shape=self.b_constant['shape'])
            self.b = tf.Variable(self.b)

    def w_var(self, shape, stddev=None):
        self.w_constant = {'shape': shape, 'stddev': stddev}

    def b_var(self, shape):
        self.b_constant = {'shape': shape}

    def set_w(self, weights):
        self.w = weights

    def set_b(self, bias):
        self.b = bias

    def Wx_plus_b(self, inX, w, b):
        with tf.name_scope('WX_b'):
            return tf.matmul(inX, w) + b

    def finish(self):
        with tf.name_scope(self.name):
            if self.w is None:
                self._w_init()
            if self.b is None:
                self._b_init()

            self.layer = self.Wx_plus_b(self.input, self.w, self.b)

            if self.isRelu:
                with tf.name_scope('Relu'):
                    self.layer = tf.nn.relu(self.layer)

            if self.keep_prob is not None:
                with tf.name_scope('dropout'):
                    self.layer = tf.nn.dropout(self.layer, self.keep_prob)


class LayerConv(object):
    def __init__(self, name, inputTensor, isRelu):
        self.name = name
        self.input = inputTensor
        self.isRelu = isRelu
        #self.keep_prob = keep_prob

        self.w = None
        self.b = None
        self.layer = None
        self.w_constant = None
        self.b_constant = None
        self.conv_constant = None
        self.max_pool_constant = None
        return

    def _w_init(self):
        with tf.name_scope('W'):
            if self.w_constant['stddev'] is not None:
                self.w = tf.truncated_normal(shape=self.w_constant['shape'],
                                             stddev=self.w_constant['stddev'])
            else:
                self.w = tf.truncated_normal(shape=self.w_constant['shape'])
            self.w = tf.Variable(self.w)

    def _b_init(self):
        with tf.name_scope('B'):
            self.b = tf.constant(0.1, shape=self.b_constant['shape'])
            self.b = tf.Variable(self.b)

    def w_var(self, shape, stddev=None):
        self.w_constant = {'shape': shape, 'stddev': stddev}

    def b_var(self, shape):
        self.b_constant = {'shape': shape}

    def set_w(self, weights):
        self.w = weights

    def set_b(self, bias):
        self.b = bias

    def conv_2d_var(self, strides=None, padding='SAME'):
        self.conv_constant = {'strides': strides, 'padding': padding}

    def max_pool_2x2_var(self, ksize=None, strides=None, padding='SAME'):
        self.max_pool_constant = {'ksize': ksize, 'strides': strides, 'padding': padding}

    def _conv_2d(self, x, W, strides=None, padding='SAME'):
        if strides is None:
            strides = [1, 1, 1, 1]
        with tf.name_scope('conv2d'):
            return tf.nn.conv2d(x, W, strides=strides, padding=padding)

    def _max_pool_2x2(self, x, ksize=None, strides=None, padding='SAME'):
        if strides is None:
            strides = [1, 2, 2, 1]
        if ksize is None:
            ksize = [1, 2, 2, 1]
        with tf.name_scope('maxPool'):
            return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding)

    def _W_conv_X_plus_b(self, inX, w, b):
        with tf.name_scope('WcX_b'):
            return self._conv_2d(inX, w,
                                 strides=self.conv_constant['strides'],
                                 padding=self.conv_constant['padding']) + b

    def finish(self):
        with tf.name_scope(self.name):
            if self.w is None:
                self._w_init()
            if self.b is None:
                self._b_init()

            self.layer = self._W_conv_X_plus_b(self.input, self.w, self.b)

            if self.isRelu:
                with tf.name_scope('Relu'):
                    self.layer = tf.nn.relu(self.layer)

            '''if self.keep_prob is not None:
                with tf.name_scope('dropout'):
                    self.layer = tf.nn.dropout(self.layer, self.keep_prob)'''


class Graph(object):
    def __init__(self, graphName):
        self.graphName = graphName
        self.countLayer = 0
        self.layerList = []
        self.layerList_kind = []

        self.softmax_out = None
        self.loss = None

        self.optimizer = None
        self.learning_rate = None

        return

    def add_LayerNormal(self, inputT, isRelu, keep_prob=None):
        self.countLayer += 1
        layerName = "Layer%02d" % self.countLayer
        Layer = LayerNormal(layerName, inputT, isRelu, keep_prob)
        self.layerList.append(Layer)
        self.layerList_kind.append("Normal")

        return Layer

    def add_LayerConv(self, inputT, isRelu, keep_prob=None):
        self.countLayer += 1
        layerName = "Layer%02d" % self.countLayer
        Layer = LayerConv(layerName, inputT, isRelu)
        self.layerList.append(Layer)
        self.layerList_kind.append("Conv")

        return Layer

    def softmax(self):
        with tf.name_scope("softmax"):
            self.softmax_out = tf.nn.softmax(self.layerList[len(self.layerList) - 1].layer)
        return self.softmax_out

    def cross_entropy(self, loss_beta_w, loss_beta_b, train_labels):
        with tf.name_scope('cross_entropy'):
            l2_loss_w = 0
            l2_loss_b = 0

            if loss_beta_w is not None:
                for ii in range(0, self.countLayer):
                    l2_loss_w += tf.nn.l2_loss(self.layerList[ii].w)
                l2_loss_w = loss_beta_w * l2_loss_w

            if loss_beta_b is not None:
                for ii in range(0, self.countLayer):
                    l2_loss_b += tf.nn.l2_loss(self.layerList[ii].b)
                l2_loss_b = loss_beta_b * l2_loss_b

            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=train_labels,
                                                        logits=self.layerList[len(self.layerList) - 1].layer))
            self.loss += l2_loss_w + l2_loss_b
        return self.loss

    def def_train_Layer(self, mGraph, layerCount, inputTensor, layer_kind):
        return TrainLayer(mGraph, layerCount, inputTensor, layer_kind)

    def train(self, needDecay, starter_learning_rate, kind_optimizer, deltaRate=None, deltaStep=None):
        with tf.name_scope('train'):
            if needDecay:
                global_step = tf.Variable(0, trainable=False)
                self.learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                                global_step, deltaStep,
                                                                deltaRate,
                                                                staircase=True)
                if kind_optimizer == "GradientDescentOptimizer":
                    self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)\
                                        .minimize(self.loss,
                                                  global_step=global_step)
                elif kind_optimizer == "AdamOptimizer":
                    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)\
                                        .minimize(self.loss,
                                                  global_step=global_step)

            elif not needDecay:
                self.learning_rate = starter_learning_rate

                if kind_optimizer == "GradientDescentOptimizer":
                    self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
                elif kind_optimizer == "AdamOptimizer":
                    self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            else:
                print("needDecay isn't Bool")
                sys.exit()

    def test_logits(self, name, inputTensor):
        with tf.name_scope(name):

            layerList = []
            testCountLayer = 0

            for ii in range(0, self.countLayer):
                testCountLayer += 1
                layerName = "%s_Layer%02d" % (name, testCountLayer)

                if ii == 0:
                    mInputTensor = inputTensor
                else:
                    if self.layerList_kind[ii - 1] == "Conv" and self.layerList_kind[ii] == "Normal":
                        tmp_layer = layerList[len(layerList) - 1].layer
                        shape = tmp_layer.get_shape().as_list()
                        mInputTensor = tf.reshape(tmp_layer, [shape[0], shape[1] * shape[2] * shape[3]])
                    else:
                        mInputTensor = layerList[ii - 1].layer

                if self.layerList_kind[ii] == "Normal":
                    layer = LayerNormal(layerName, mInputTensor,
                                        self.layerList[ii].isRelu,
                                        self.layerList[ii].keep_prob)

                    layer.set_w(self.layerList[ii].w)
                    layer.set_b(self.layerList[ii].b)
                    layer.finish()

                elif self.layerList_kind[ii] == "Conv":
                    layer = LayerConv(layerName, mInputTensor,
                                      self.layerList[ii].isRelu)

                    layer.set_w(self.layerList[ii].w)
                    layer.set_b(self.layerList[ii].b)
                    layer.conv_2d_var(strides=self.layerList[ii].conv_constant['strides'],
                                      padding=self.layerList[ii].conv_constant['padding'])
                    layer.finish()
                else:
                    print("layer_kind is error.\nNeed Normal or Conv")
                    sys.exit()

                layerList.append(layer)

            prediction = tf.nn.softmax(layerList[len(layerList) - 1].layer)

        return layerList, prediction


class TrainLayer(object):
    def __init__(self, mGraph, layerCount, inputTensor, layer_kind):
        self.graph = mGraph
        self.layerCount = layerCount
        self.inputTensor = inputTensor
        self.layer_kind = layer_kind

        self.layerList = []

        self.layer_isRelu = None
        self.layer_keep_prob = None
        self.stddev = None

        self.layer_input_dim = None
        self.layer_output_dim = None

        return

    def set_LayerVar(self, layer_isRelu, layer_keep_prob, stddev):
        self.layer_isRelu = layer_isRelu
        self.layer_keep_prob = layer_keep_prob
        self.stddev = stddev

    def set_LayerSize(self, layer_input_dim, layer_output_dim):
        self.layer_input_dim = layer_input_dim
        self.layer_output_dim = layer_output_dim

    def finish(self):
        for ii in range(0, self.layerCount):
            if ii == 0:
                mInputTensor = self.inputTensor
            else:
                if self.layer_kind[ii - 1] == "Conv" and self.layer_kind[ii] == "Normal":
                    tmp_layer = self.layerList[len(self.layerList) - 1].layer
                    shape = tmp_layer.get_shape().as_list()
                    mInputTensor = tf.reshape(tmp_layer, [shape[0], shape[1] * shape[2] * shape[3]])
                else:
                    mInputTensor = self.layerList[ii - 1].layer

            if self.layer_kind[ii] == "Normal":
                layer = self.graph.add_LayerNormal(mInputTensor,
                                                   self.layer_isRelu[ii],
                                                   self.layer_keep_prob(ii))

                layer.w_var(shape=self.layer_input_dim[ii], stddev=self.stddev)
                layer.b_var(shape=self.layer_output_dim[ii])
                layer.finish()

            elif self.layer_kind[ii] == "Conv":
                layer = self.graph.add_LayerConv(mInputTensor,
                                                 self.layer_isRelu[ii],
                                                 self.layer_keep_prob(ii))

                layer.w_var(shape=self.layer_input_dim[ii], stddev=self.stddev)
                layer.b_var(shape=self.layer_output_dim[ii])
                layer.conv_2d_var(strides=[1, 2, 2, 1], padding='SAME')
                layer.finish()
            else:
                print("layer_kind is error.\nNeed Normal or Conv")
                sys.exit()

            self.layerList.append(layer)
        return self.layerList
