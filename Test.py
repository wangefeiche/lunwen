import numpy as np 
from Environment import Environment
from Tomato import Tomato

def Test():
    env = Environment()
    observation = env.reset()
    done = False
    segmentcount = 0
    while not done:
        # choose action
        # print(segmentcount, observation)
        # bitrate = 9
        bitrate = Tomato(observation[0], observation[1], observation[2])
        
        observation_, reward, done = env.step(bitrate)

        observation = observation_

        segmentcount += 1

    # print(env.bitrate_record)
    env.th_plot()
    env.buffer_plot()
    env.log_output()
    # print(env.reward_record)

if __name__ == "__main__":
    Test()
    

    