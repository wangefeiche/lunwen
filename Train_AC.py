import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
from Environment import Environment
from ActorCritic import Actor, Critic
from PolicyGradient import PolicyGradient

tf.set_random_seed(2)  # reproducible
# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 1000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 100   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.001     # learning rate for critic

env = Environment()
N_F = env.n_features
N_A = env.n_actions


def Train():
    sess = tf.Session()

    actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
    critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

    sess.run(tf.global_variables_initializer())

    reward_history = [['episode', 'ave_reward']]

    

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)

    for i_episode in range(MAX_EPISODE):
        env.__init__()
        s = env.reset()
        t = 0
        track_r = []
        buffer_s, buffer_a, buffer_r, buffer_v_target = [], [], [], []
        while True:
            # buffer_s, buffer_a, buffer_r, buffer_v_target = [], [], [], []
            a = actor.choose_action(s)
            s_, r, done, r_penalty = env.step(s, a)
            track_r.append(r)
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append(r_penalty)

            # if done:
            #     v_s_ = r
            # else:
                # v_s_ = r + GAMMA * sess.run(critic.v, {critic.s: s_[np.newaxis, :]})[0, 0]
            # buffer_v_target.append(v_s_)
            if done:
                v_s_ = sum(track_r)
                for r in buffer_r[::-1]:  # reverse buffer r
                    v_s_ = r + GAMMA * v_s_
                    buffer_v_target.append(v_s_)
                buffer_v_target.reverse()
                buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                # print(buffer_s)
                # print(buffer_s.shape, buffer_a.shape, buffer_v_target.shape)
                td_error = critic.learn(buffer_s, buffer_v_target)  # gradient = grad[r + gamma * V(s_) - V(s)]
                actor.learn(buffer_s, buffer_a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]
            # buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
            #
            # td_error = critic.learn(buffer_s, buffer_v_target)  # gradient = grad[r + gamma * V(s_) - V(s)]
            # actor.learn(buffer_s, buffer_a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

            s = s_
            t += 1

            if done or t >= MAX_EP_STEPS:
                break

        print(i_episode, "  reward: ", sum(track_r)/len(track_r))
        bitrate_list = [i / 1e7 for i in env.bitrate_record]
        print(bitrate_list)
        reward_history.append([i_episode, sum(track_r)/len(track_r)])
    env.th_plot()
    env.buffer_plot()
    env.log_output()
    import csv
    with open('reward_history.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(reward_history)
        
if __name__ == "__main__":
    Train()
    y = pd.read_csv('reward_history.csv', header=0, index_col=0).values
    y = y.reshape(y.shape[0]).tolist()
    x = range(len(y))
    plt.figure(figsize=(40,24))
    plt.subplot(111)
    plt.plot(x, y,'r-',label = 'reward',linewidth=2.0)
    plt.grid(True)
    plt.xlabel("Time/s", fontsize=20)
    plt.legend()
    plt.savefig("reward_history.png")
