"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import time
import csv
import tensorflow as tf
import math

np.random.seed(3)  # reproducible

data_file = "sim0_cl0_throughputLog.txt"
qtable_file = "qtable.txt"
N_STATES = 15   # the length of the 1 dimensional world
ACTIONS = ['1', '2','3','4','5','6','7','8','9','10']     # available actions
EPSILON = 0.8   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 10000   # maximum episodes
FRESH_TIME = 0.01    # fresh time for one move
bufferLength = 0 # the client buffer length
downloadStart = 0
downloadEnd = 0
segmentDuration = 1

SegmentSize_360s_list = []
with open("SegmentSize_360s.txt",'r') as SegmentSize_360s_readfile:
    n=0
    while True:
        lines = SegmentSize_360s_readfile.readline() # 整行读取数据
        if not lines:
            break
        i = lines.split()
        SegmentSize_360s_list.append([float(x) for x in i])
        n = n+1
#print(SegmentSize_360s_list)
SegmentSize_360s_list = [[float(x) for x in row] for row in SegmentSize_360s_list]

DlRxPhyStats_time, DlRxPhyStats_tbsize = [], []
with open(data_file,'r') as DlRxPhyStats_to_read:
    n=0
    while True:
        lines = DlRxPhyStats_to_read.readline() # 整行读取数据
        if not lines:
            break
        
        i = lines.split()
        DlRxPhyStats_time.append(float(i[0]))
        DlRxPhyStats_tbsize.append(float(i[1])*8)
        n = n+1

DlRsrpSinrStats_time, DlRsrpSinrStats_rsrp = [], []
with open("DlRsrpSinrStats.txt",'r') as DlRsrpSinrStats_to_read:
    n=0
    while True:
        lines = DlRsrpSinrStats_to_read.readline() # 整行读取数据
        if not lines:
            break
        if n == 0:
            pass
        else:
            i = lines.split()
            DlRsrpSinrStats_time.append(float(i[0]))
            DlRsrpSinrStats_rsrp.append(float(i[4]))
        n = n+1

DlRsrpSinrStats_rsrp = [10*math.log10(1000*x) for x in DlRsrpSinrStats_rsrp]

#print(DlRxPhyStats_time)

class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        #self.saver.restore(self.sess, "model/model.ckpt")
        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        #w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        w_initializer, b_initializer = tf.random_normal_initializer(0.,0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, tf.nn.relu, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.q_next = tf.layers.dense(t1, self.n_actions, tf.nn.relu, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[None, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        #print("  S: ",observation," actions_value: ",actions_value)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        
        _, cost, q_target_temp, q_eval_wrt_a_temp, q_next_temp= self.sess.run(
            [self._train_op, self.loss,self.q_target,self.q_eval_wrt_a,self.q_next],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })
        #print(q_target_temp)
        #print(q_eval_wrt_a_temp)
        #print(q_next_temp)
        #print(cost)
        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        print(self.learn_step_counter,self.epsilon,cost)

    def plot_cost(self):
        self.saver.save(self.sess, "model/model.ckpt")
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

def run_maze():
    step = 0
    for episode in range(10000):
        # initial observation
        observation = np.array([-92.4,-92.3,0])
        segment_Count = 0
        T = 0
        B = 0
        while True:
            # fresh env
            #update_env(observation[0], episode, step)

            # RL choose action based on observation
            action = RL.choose_action(observation)
            # RL take action and get next observation and reward
            observation_, T,B,reward, done = get_env_feedback(observation,T,B,segment_Count,action)
            print(step,observation,action,reward,segment_Count)
            segment_Count += 1
            RL.store_transition(observation, action, reward, observation_)

            if (step > 50) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

        print("episode: ",episode)

    # end of game
    print('game over')


def get_env_feedback(S,T,B,SC,A):
    # This is how agent will interact with the environment
    action = int(A)
    segmentSize = SegmentSize_360s_list[action][SC]*segmentDuration*8
    downloadStart = DlRxPhyStats_time[T]
    downloadEnd = 0
    size_sum = 0
    T_ = T
    B_ = B
    R = 0
    for data_size in DlRxPhyStats_tbsize[T:]:
        if size_sum < segmentSize:
            size_sum = size_sum + data_size
            T_ = T_ + 1
        else:
            downloadEnd = DlRxPhyStats_time[T_]
            break
    interval = 1
    rsrp_data = []
    start_time = downloadEnd-2
    flag = 0
    sum_rsrp = 0
    count = 0
    #print(DlRxPhyStats_time)
    for rtime,rsrp in zip(DlRsrpSinrStats_time,DlRsrpSinrStats_rsrp):
        if rtime >= start_time and flag < 2:
            if rtime < start_time+interval:
                sum_rsrp += rsrp
                count += 1
            else:
                flag += 1
                start_time = start_time+interval
                rsrp_data.append(sum_rsrp/count)
                sum_rsrp = 0
                count = 0
    S_1 = np.array(rsrp_data)
    
    #print("===============",downloadStart,downloadEnd)
    if SC == 0:
        B_ = B_ + segmentDuration
    else:
        B_ = B_ + segmentDuration - (downloadEnd-downloadStart)

    S_ = np.append(S_1,B_)
    if B_ > 1 and B_ < 2 :
        if T_ == len(DlRxPhyStats_tbsize):
            R = 2
            done = True
        else:
            R = 0
            done = False
    else:
        if SC == 0 and downloadEnd-downloadStart < 3 and downloadEnd-downloadStart > 0.8:
            R = 0
            done = False
        else:
            R = -1
            done = True
     
    return S_,T_,B_,R,done



def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S ==  np.array[N_STATES-1,-1]:
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S[0]] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

if __name__ == "__main__":
    RL = DeepQNetwork(10,3,
                      learning_rate=0.001,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=1000,
                      #output_graph=True
                      )
    
    run_maze()
    RL.plot_cost()

