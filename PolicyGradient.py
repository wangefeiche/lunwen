import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# reproducible
# np.random.seed(1)
# tf.set_random_seed(1)

np.random.seed(3)
tf.set_random_seed(3)

class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.learn_step_counter = 0
        self.output_graph = output_graph

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.merge = tf.summary.merge_all()
        if self.output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            self.writer = tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        # self.saver.restore(self.sess, "model/DQN_model.ckpt")


    def get_weight(self, shape,lamda):
    
        var = tf.Variable(tf.random_normal(shape=shape),dtype=tf.float32)
    
        tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(lamda)(var))

        return var



    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        """ with tf.variable_scope('eval_net'):
            layer_dimension = [self.n_features,64,128,256,self.n_actions]

            n_layers = len(layer_dimension)

            cur_layer = self.tf_obs
            
            in_dimension = layer_dimension[0]
            
            for i in range(1,n_layers):
                
                out_dimension = layer_dimension[i]
                
                weight = self.get_weight([in_dimension,out_dimension],0.001)
                
                bias = tf.Variable(tf.constant(0.1,shape=[out_dimension]))
                
                cur_layer = tf.nn.relu(tf.matmul(cur_layer,weight) + bias)

                # if i == 1:
                #     self.test_var = bias
                
                in_dimension = layer_dimension[i]

            self.all_act_prob = tf.nn.softmax(cur_layer, name='act_prob') """

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.03), tf.constant_initializer(0.01)
        with tf.variable_scope('eval_net'):
            # e_ue_1 = tf.layers.dense(self.tf_obs, 256, tf.nn.relu, kernel_initializer=w_initializer,
            #                      bias_initializer=b_initializer, name='e_ue_1')
            # e_ue_2 = tf.layers.dense(e_ue_1, 256, tf.nn.relu, kernel_initializer=w_initializer,
            #                      bias_initializer=b_initializer, name='e_ue_2')
            # e_ue_3 = tf.layers.dense(e_ue_2, 128, tf.nn.relu, kernel_initializer=w_initializer,
            #                      bias_initializer=b_initializer, name='e_ue_3')
            # e_at_1 = tf.layers.dense(self.at_s, 32, tf.nn.relu, kernel_initializer=w_initializer,
            #                      bias_initializer=b_initializer, name='e_at_1')
            # e_at_2 = tf.layers.dense(e_at_1, 64, tf.nn.relu, kernel_initializer=w_initializer,
            #                      bias_initializer=b_initializer, name='e_at_2')
            e_1 = tf.layers.dense(self.tf_obs, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e_1')
            e_2 = tf.layers.dense(e_1, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e_2')
            e_3 = tf.layers.dense(e_2, 256, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e_3')
            all_act = tf.layers.dense(e_3, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q_eval')
            all_act_prob = tf.nn.softmax(all_act, name='act_prob')
            self.all_act = e_1
            self.all_act_prob = all_act_prob
        

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob + 1e-10)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            c_loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss
            # tf.add_to_collection("losses",c_loss)
            # loss = tf.add_n(tf.get_collection("losses"))
            self.loss = c_loss
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        # print(test_var)
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        # print(observation, action)
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        # print(discounted_ep_rs_norm)
        # train on episode
        _, summary, cost = self.sess.run([self.train_op, self.merge, self.loss], feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, action_length]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })
        print("loss: ", cost)
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        if self.output_graph:
            self.writer.add_summary(summary, self.learn_step_counter)
        self.learn_step_counter += 1
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        # print(discounted_ep_rs)
        # normalize episode rewards
        if len(self.ep_rs) > 1:
            discounted_ep_rs -= np.mean(discounted_ep_rs)
            discounted_ep_rs /= np.std(discounted_ep_rs)
        
        return discounted_ep_rs

    def plot_cost(self):
        self.saver.save(self.sess, "model/DQN_model.ckpt")

if __name__ == '__main__':
    RL = PolicyGradient(495, 14400,
                        learning_rate=0.01,
                        reward_decay=0.9,
                        output_graph=True
                        )