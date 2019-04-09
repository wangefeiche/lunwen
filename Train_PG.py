"""
Train.py 是训练时运行的脚本
test.py是测试时运行的脚本
"""
from Environment import Environment
from PolicyGradient import PolicyGradient
import csv
import numpy as np

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = Environment()

RL = PolicyGradient(
    n_actions=env.n_actions,
    n_features=env.n_features,
    learning_rate=0.001,
    reward_decay=0.99,
    output_graph=False,
)

for i_episode in range(3000):
    env.__init__()
    observation = env.reset()

    while True:

        action = RL.choose_action(observation)

        observation_, reward, done = env.step(action)

        RL.store_transition(observation, action, reward)

        if done:
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            print("episode:", i_episode, "  reward:", running_reward)

            vt = RL.learn()

            
            break

        observation = observation_
    env.th_plot()
    env.buffer_plot()
    env.log_output()