import numpy as np
import tensorflow as tf
import preprocess


INPUT_NODES = 0
OUTPUT_NODES = 0
LAYER1_NODE = 100
BATCH_SIZE = 20
LEARNING_RATE_BASE = 0.9    # 0.8
LEARNING_RATE_DECAY = 0.99  # 0.99
REGULARIZATION_RATE = 0.00002


def train(x_train, y_train, x_test, y_test):
    x = tf.placeholder(tf.float32, shape=(None, INPUT_NODES), name='x-input')
    y_ = tf.placeholder(tf.float32, shape=(None, OUTPUT_NODES), name='y-input')

    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODES, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODES], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODES]))

    layer1 = tf.nn.relu(tf.matmul(x, weights1)+biases1)
    y = tf.matmul(layer1, weights2)+biases2
    # y = tf.Print(y, [y], 'Current Result:')

    # 计算欧氏距离
    euclidean = tf.reduce_mean(tf.square(y-y_))

    # 正则器
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)

    loss = euclidean + regularization
    # 设置指数衰减的学习率
    learning_rate = LEARNING_RATE_BASE
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    train_op = tf.group(train_step)

    # 分类器
    logits = tf.layers.dense(y, 4, name='fc2')
    y_pred_cls = tf.argmax(tf.nn.softmax(logits), 1)  # 预测类别

    # 准确率
    correct_pred = tf.equal(tf.argmax(y_, 1), y_pred_cls)
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.Session() as sess:
        print('Staring Train  x:', x_train.shape)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(len(x_train)//BATCH_SIZE):
            xs = x_train[BATCH_SIZE * i:(i + 1) * BATCH_SIZE]
            ys = y_train[BATCH_SIZE * i:(i + 1) * BATCH_SIZE]
            sess.run(train_op, feed_dict={x: xs, y_: ys})
            loss_val, acc_val = sess.run([loss,acc], feed_dict={x:x_test, y_:y_test})
            print('Acc :%.2f'%(acc_val*100), '%  Loss :', loss_val)

def main(argv=None):
    global INPUT_NODES, OUTPUT_NODES
    x, y = preprocess.getdata(r'C:\Users\chenshuai\Documents\材料学院\贝氏体钢数据统计-chenshuai.xlsx')
    INPUT_NODES = x.shape[1]
    OUTPUT_NODES = y.shape[1]
    x_train = x[:-80, :]
    y_train = y[:-80, :]
    x_test = x[-80:, :]
    y_test = y[-80:, :]
    # print(x_train[0])
    # print(y_train[0])
    train(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    tf.app.run()