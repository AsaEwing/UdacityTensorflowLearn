import sys
import math
import time

import numpy as np
import tensorflow as tf


def getSpace(count):
    return " " * count


class tf_Constant(object):
    def __init__(self):
        print("\n~ # Create Init Need")

        self.randomCount = 5000
        self.trainRate = 0.2
        self.trainStep = 1000000
        self.errorAgree = 0.002

        print("    =====>  randomCount : %15.4f" % self.randomCount)
        print("    =====>  trainRate   : %15.4f" % self.trainRate)
        print("    =====>  trainStep   : %15.4f" % self.trainStep)
        print("    =====>  errorAgree  : %15.4f" % self.errorAgree)
        return


# TODO @ create train data #
class tf_TrainData(tf_Constant):
    def __init__(self):
        super().__init__()
        print("\n~ # Create Train Data")

        self.init_Weights = [0.1, 0, 2]
        self.init_Biases = [-1.8]
        self.x_trainData = np.random.rand(self.randomCount).astype(np.float32)
        self.y_trainData = self.init_Biases[0] \
                           + (self.x_trainData ** 1) * self.init_Weights[0] \
                           + (self.x_trainData ** 2) * self.init_Weights[1] \
                           + (self.x_trainData ** 3) * self.init_Weights[2]
        print("  # train data:")
        print("    =====>  y_trainData = init_Biases[0]"
              "\n%s+ (x_trainData ** 1) * init_Weights[0]"
              "\n%s+ (x_trainData ** 2) * init_Weights[1]"
              "\n%s+ (x_trainData ** 3) * init_Weights[2]"
              % (getSpace(26), getSpace(26), getSpace(26)))
        print("    =====>  init_Weights    = %s" % self.init_Weights)
        print("    =====>  init_Biases     = %s" % self.init_Biases)
        print("    =====>  max x_trainData = %15.11f" % max(self.x_trainData))
        print("    =====>  min x_trainData = %15.11f" % min(self.x_trainData))
        print("    =====>  max y_trainData = %15.11f" % max(self.y_trainData))
        print("    =====>  min y_trainData = %15.11f" % min(self.y_trainData))
        return


class tf_Structure(tf_TrainData):
    def __init__(self):
        super().__init__()
        print("\n~ # Create Train Structure")

        self.Weights = tf.Variable(tf.random_uniform([3], -2.0, 2.0))
        self.biases = tf.Variable(tf.zeros([1]))
        self.y_trainFunction = self.biases.value()[0] \
                               + (self.x_trainData ** 1) * self.Weights.value()[0] \
                               + (self.x_trainData ** 2) * self.Weights.value()[1] \
                               + (self.x_trainData ** 3) * self.Weights.value()[2]
        print("  # train function:")
        print("    =====>  y_trainFunction = biases[0]"
              "\n%s+ (x_trainData ** 1) * Weights[0]"
              "\n%s+ (x_trainData ** 2) * Weights[1]"
              "\n%s+ (x_trainData ** 3) * Weights[2]"
              % (getSpace(30), getSpace(30), getSpace(30)))
        self.trainLoss = tf.reduce_mean(tf.square(self.y_trainFunction - self.y_trainData))

        self.optimizer = tf.train.GradientDescentOptimizer(self.trainRate)
        self.train = self.optimizer.minimize(self.trainLoss)
        return


# TODO @ 程式初始化 ＃
tfS = tf_Structure()

Time_AllStart = time.clock() * 1000
print("~ # test01 程式開始")

# TODO @ init #
print("\n~ # Init")
init = tf.initialize_all_variables()  # tf 马上就要废弃这种写法
# init = tf.global_variables_initializer()  # 替换成这样就好
sess = tf.Session()
sess.run(init)  # Very important

output_Weights = [0, 0, 0]
output_Biases = [0]
trainIsOk = False

# TODO @ train #
print("\n~ # Train Start")
tmpStr = "  # %7d # Weight[0]:[%21.17f]  err : %21.17f" \
         "\n%sWeight[1]:[%21.17f]  err : %21.17f" \
         "\n%sWeight[2]:[%21.17f]  err : %21.17f" \
         "\n%sBiases[0]:[%21.17f]  err : %21.17f"

now_Weights = [0, 0, 0]
now_Biases = [0]
before_Weights = [0, 0, 0]
before_Biases = [0]

equal_Count = 0
equal_List = []
not_equal_Count = 0
small_change_Count = 0
c_w0 = 999
c_w1 = 999
c_w2 = 999
c_b0 = 999

c2_w0 = 999
c2_w1 = 999
c2_w2 = 999
c2_b0 = 999

c3_w0 = 999
c3_w1 = 999
c3_w2 = 999
c3_b0 = 999

for step in range(0, tfS.trainStep + 1):
    sess.run(tfS.train)
    now_Weights = list(sess.run(tfS.Weights))
    now_Biases = list(sess.run(tfS.biases))

    if math.isnan(now_Weights[0]) \
            or math.isnan(now_Weights[1]) \
            or math.isnan(now_Weights[2]) \
            or math.isnan(now_Biases[0]):
        print("\n  ====> Step nan error :", step)
        break
    elif step != 0:
        c_w0 = abs(now_Weights[0] - before_Weights[0])
        c_w1 = abs(now_Weights[1] - before_Weights[1])
        c_w2 = abs(now_Weights[2] - before_Weights[2])
        c_b0 = abs(now_Biases[0] - before_Biases[0])

        c2_w0 = abs(c_w0 / before_Weights[0]) * 1000000
        c2_w1 = abs(c_w1 / before_Weights[1]) * 1000000
        c2_w2 = abs(c_w2 / before_Weights[2]) * 1000000
        c2_b0 = abs(c_b0 / before_Biases[0]) * 1000000

        c3_w0 = abs(now_Weights[0] - tfS.init_Weights[0])
        c3_w1 = abs(now_Weights[1] - tfS.init_Weights[1])
        c3_w2 = abs(now_Weights[2] - tfS.init_Weights[2])
        c3_b0 = abs(now_Biases[0] - tfS.init_Biases[0])

    if step % 10000 == 0:
        print("\n", tmpStr % (step, now_Weights[0], c3_w0,
                              getSpace(15), now_Weights[1], c3_w1,
                              getSpace(15), now_Weights[2], c3_w2,
                              getSpace(15), now_Biases[0], c3_b0))

    if c2_w0 < 1 and c2_w1 < 1 and c2_w2 < 1 and c2_b0 < 1:
        small_change_Count += 1
        if small_change_Count == 1:
            print("\n  # ==> Step small change start :", step)
            print(tmpStr % (step, now_Weights[0], c3_w0,
                            getSpace(15), now_Weights[1], c3_w1,
                            getSpace(15), now_Weights[2], c3_w2,
                            getSpace(15), now_Biases[0], c3_b0))

    if c_w0 == 0 and c_w1 == 0 and c_w2 == 0 and c_b0 == 0:
        equal_List.insert(equal_Count, step)
        equal_Count += 1

        if equal_Count == 1:
            print("\n  # ==> Step equal start :", step)
            print(tmpStr % (step, now_Weights[0], c3_w0,
                            getSpace(15), now_Weights[1], c3_w1,
                            getSpace(15), now_Weights[2], c3_w2,
                            getSpace(15), now_Biases[0], c3_b0))

    else:
        before_Weights = list(now_Weights)
        before_Biases = list(now_Biases)
        not_equal_Count += 1

    if (c3_w0 < tfS.errorAgree and c3_w1 < tfS.errorAgree and c3_w2 < tfS.errorAgree and c3_b0 < tfS.errorAgree) \
            and (equal_Count > 100 or small_change_Count > 1000):
        print("\n  ====> Step end :", step)
        print(tmpStr % (step, now_Weights[0], c3_w0,
                        getSpace(15), now_Weights[1], c3_w1,
                        getSpace(15), now_Weights[2], c3_w2,
                        getSpace(15), now_Biases[0], c3_b0))
        output_Weights = list(before_Weights)
        output_Biases = list(before_Biases)
        trainIsOk = True
        break

print("\n  # equal EC = %s, NEC = %s" % (equal_Count, not_equal_Count))
print("  # equal List length = %s" % len(equal_List))
print("  # small change SCC = %s" % small_change_Count)
print("~ # Train End")

# TODO @ test train
if trainIsOk:
    print("\n~ # Test Train Start")
    x_testData = np.random.rand(tfS.randomCount).astype(np.float32)
    y_testData = output_Biases[0] \
                 + (x_testData ** 1) * output_Weights[0] \
                 + (x_testData ** 2) * output_Weights[1] \
                 + (x_testData ** 3) * output_Weights[2]
    # testLoss = tf.reduce_mean(tf.square(y_testData - y_trainData))
    testLoss = sum((y_testData - tfS.y_trainData) ** 2) / len(y_testData)
    print("  # ==> Test mean square loss :", testLoss)
    print("  # ==> Test max loss :", max(abs(y_testData - tfS.y_trainData)))
    print("  # ==> Test min loss :", min(abs(y_testData - tfS.y_trainData)))
    print("\n~ # Test Train End")
else:
    print("\n~ # Test Train None")

# TODO @ 程式結束 ＃
Time_End = time.clock() * 1000
if Time_End - Time_AllStart > 3600000:
    print('~ # test01 程式結束，時間共：%08.3f h' % float((Time_End - Time_AllStart) / 3600000))
elif Time_End - Time_AllStart > 60000:
    print('~ # test01 程式結束，時間共：%08.3f min' % float((Time_End - Time_AllStart) / 60000))
elif Time_End - Time_AllStart > 1000:
    print('~ # test01 程式結束，時間共：%08.3f s' % float((Time_End - Time_AllStart) / 1000))
else:
    print('~ # test01 程式結束，時間共：%08.3f ms' % float(Time_End - Time_AllStart))
