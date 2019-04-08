import numpy as np 
import tensorflow as tf
from Environment import Environment
from ActorCritic import Actor, Critic
from PolicyGradient import PolicyGradient

tf.set_random_seed(2)  # reproducible
# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 1000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.01    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = Environment()
N_F = env.n_features
N_A = env.n_actions


def Train():
    sess = tf.Session()

    actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
    critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

    sess.run(tf.global_variables_initializer())

    

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)

    for i_episode in range(MAX_EPISODE):
        env.__init__()
        s = env.reset()
        t = 0
        track_r = []
        done = False
        while not done:

            a = actor.choose_action(s)

            s_, r, done = env.step(a)

            # if done: r = -20

            track_r.append(r)

            td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

            s = s_
            t += 1

        print("  reward: ", sum(track_r))
    env.th_plot()
    env.buffer_plot()
    env.log_output()
        
if __name__ == "__main__":
    Train()
    

    