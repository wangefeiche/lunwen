import tensorflow as tf
import numpy as np
import os

np.random.seed(2)
tf.set_random_seed(2)  # reproducible
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'


GAMMA = 0.9     # reward discount in TD error
N_A = 10

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [None, n_features], "state")
        self.a = tf.placeholder(tf.int32, [None,1], "act")
        self.td_error = tf.placeholder(tf.float32, [None,1], "td_error")  # TD_error
        w_initializer, b_initializer = tf.random_normal_initializer(0.,0.001), tf.constant_initializer(0.001)
        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=16,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=w_initializer,    # weights
                bias_initializer=b_initializer,  # biases
                name='l1'
            )

            l2 = tf.layers.dense(
                inputs=l1,
                units=32,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=w_initializer,    # weights
                bias_initializer=b_initializer,  # biases
                name='l2'
            )

            l3 = tf.layers.dense(
                inputs=l2,
                units=64,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=w_initializer,    # weights
                bias_initializer=b_initializer,  # biases
                name='l3'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l3,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=w_initializer,  # weights
                bias_initializer=b_initializer,  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob + 1e-5) * tf.one_hot(self.a, N_A, dtype=tf.float32)
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        # s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        # print(probs)
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [None, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [None, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        w_initializer, b_initializer = tf.random_normal_initializer(0.,0.003), tf.constant_initializer(0.001)
        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=16,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=w_initializer,  # weights
                bias_initializer=b_initializer,  # biases
                name='l1'
            )

            l2 = tf.layers.dense(
                inputs=l1,
                units=32,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=w_initializer,    # weights
                bias_initializer=b_initializer,  # biases
                name='l2'
            )

            l3 = tf.layers.dense(
                inputs=l2,
                units=64,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=w_initializer,    # weights
                bias_initializer=b_initializer,  # biases
                name='l3'
            )

            self.v = tf.layers.dense(
                inputs=l3,
                units=1,  # output units
                activation=None,
                kernel_initializer=w_initializer,  # weights
                bias_initializer=b_initializer,  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, v_):
        # s = s[np.newaxis, :]

        td_error, v, _ = self.sess.run([self.td_error, self.v, self.train_op],
                                          {self.s: s, self.v_: v_})
        print(s, 'v_: ', v_, ' v: ', v)

        return td_error


