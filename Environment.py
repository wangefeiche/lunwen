import numpy as np
import pandas as pd
import time
import csv
import tensorflow as tf
import math

# np.random.seed(3)  # reproducible

# data_file = "sim0_cl0_throughputLog.txt"
# qtable_file = "qtable.txt"
# N_STATES = 15   # the length of the 1 dimensional world
# ACTIONS = ['1', '2','3','4','5','6','7','8','9','10']     # available actions
# EPSILON = 0.8   # greedy police
# ALPHA = 0.1     # learning rate
# GAMMA = 0.9    # discount factor
# MAX_EPISODES = 10000   # maximum episodes
# FRESH_TIME = 0.01    # fresh time for one move
# bufferLength = 0 # the client buffer length
# downloadStart = 0
# downloadEnd = 0
# segmentDuration = 1

# SegmentSize_360s_list = []
# with open("SegmentSize_360s.txt",'r') as SegmentSize_360s_readfile:
#     n=0
#     while True:
#         lines = SegmentSize_360s_readfile.readline() # 整行读取数据
#         if not lines:
#             break
#         i = lines.split()
#         SegmentSize_360s_list.append([float(x) for x in i])
#         n = n+1
# #print(SegmentSize_360s_list)
# SegmentSize_360s_list = [[float(x) for x in row] for row in SegmentSize_360s_list]

# DlRxPhyStats_time, DlRxPhyStats_tbsize = [], []
# with open(data_file,'r') as DlRxPhyStats_to_read:
#     n=0
#     while True:
#         lines = DlRxPhyStats_to_read.readline() # 整行读取数据
#         if not lines:
#             break
        
#         i = lines.split()
#         DlRxPhyStats_time.append(float(i[0]))
#         DlRxPhyStats_tbsize.append(float(i[1])*8)
#         n = n+1

# DlRsrpSinrStats_time, DlRsrpSinrStats_rsrp = [], []
# with open("DlRsrpSinrStats.txt",'r') as DlRsrpSinrStats_to_read:
#     n=0
#     while True:
#         lines = DlRsrpSinrStats_to_read.readline() # 整行读取数据
#         if not lines:
#             break
#         if n == 0:
#             pass
#         else:
#             i = lines.split()
#             DlRsrpSinrStats_time.append(float(i[0]))
#             DlRsrpSinrStats_rsrp.append(float(i[4]))
#         n = n+1

# DlRsrpSinrStats_rsrp = [10*math.log10(1000*x) for x in DlRsrpSinrStats_rsrp]

# #print(DlRxPhyStats_time)




# def get_env_feedback(S,T,B,SC,A):
#     # This is how agent will interact with the environment
#     action = int(A)
#     segmentSize = SegmentSize_360s_list[action][SC]*segmentDuration*8
#     downloadStart = DlRxPhyStats_time[T]
#     downloadEnd = 0
#     size_sum = 0
#     T_ = T
#     B_ = B
#     R = 0
#     for data_size in DlRxPhyStats_tbsize[T:]:
#         if size_sum < segmentSize:
#             size_sum = size_sum + data_size
#             T_ = T_ + 1
#         else:
#             downloadEnd = DlRxPhyStats_time[T_]
#             break
#     interval = 1
#     rsrp_data = []
#     start_time = downloadEnd-2
#     flag = 0
#     sum_rsrp = 0
#     count = 0
#     #print(DlRxPhyStats_time)
#     for rtime,rsrp in zip(DlRsrpSinrStats_time,DlRsrpSinrStats_rsrp):
#         if rtime >= start_time and flag < 2:
#             if rtime < start_time+interval:
#                 sum_rsrp += rsrp
#                 count += 1
#             else:
#                 flag += 1
#                 start_time = start_time+interval
#                 rsrp_data.append(sum_rsrp/count)
#                 sum_rsrp = 0
#                 count = 0
#     S_1 = np.array(rsrp_data)
    
#     #print("===============",downloadStart,downloadEnd)
#     if SC == 0:
#         B_ = B_ + segmentDuration
#     else:
#         B_ = B_ + segmentDuration - (downloadEnd-downloadStart)

#     S_ = np.append(S_1,B_)
#     if B_ > 1 and B_ < 2 :
#         if T_ == len(DlRxPhyStats_tbsize):
#             R = 2
#             done = True
#         else:
#             R = 0
#             done = False
#     else:
#         if SC == 0 and downloadEnd-downloadStart < 3 and downloadEnd-downloadStart > 0.8:
#             R = 0
#             done = False
#         else:
#             R = -1
#             done = True
     
#     return S_,T_,B_,R,done



# def update_env(S, episode, step_counter):
#     # This is how environment be updated
#     env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
#     if S ==  np.array[N_STATES-1,-1]:
#         interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
#         print('\r{}'.format(interaction), end='')
#         time.sleep(2)
#         print('\r                                ', end='')
#     else:
#         env_list[S[0]] = 'o'
#         interaction = ''.join(env_list)
#         print('\r{}'.format(interaction), end='')
#         time.sleep(FRESH_TIME)

# if __name__ == "__main__":
#     RL = DeepQNetwork(10,3,
#                       learning_rate=0.001,
#                       reward_decay=0.9,
#                       e_greedy=0.9,
#                       replace_target_iter=200,
#                       memory_size=1000,
#                       #output_graph=True
#                       )
    
#     run_maze()
#     RL.plot_cost()



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
with open("sim0_cl0_throughputLog.txt",'r') as DlRxPhyStats_to_read:
    n=0
    while True:
        lines = DlRxPhyStats_to_read.readline() # 整行读取数据
        if not lines:
            break
        
        i = lines.split()
        DlRxPhyStats_time.append(float(i[0]))
        DlRxPhyStats_tbsize.append(float(i[1])*8)
        n = n+1

class Environment():
    def __init__(self):
        self.action_space = ['0','1','2','3','4','5','6','7','8','9']
        self.n_actions = len(self.action_space)
        self.n_features = 3
        self.segmentDuration = 1
        self.segmentcount = 0
        self.network_trace_time = DlRxPhyStats_time
        self.network_trace_size = DlRxPhyStats_tbsize
        self.video_trace = SegmentSize_360s_list
        self.bitrate_record = []
        self.segment_dltime_list = []
        self.buffer_list = [0]
        self.rebuffer_starttime_list = []
        self.rebuffer_endtime_lsit = []
        self.tb_count = 0
        self.s = 0

    def reset(self):
        origin = np.array([0,0,0])

        return origin

    def step(self, action):
        # s = 0
        s_ = self.s
        reward = 0
        done = True

        action = int(action)
        segmentSize = self.video_trace[action][self.segmentcount]*self.segmentDuration*8
        downloadStart = self.network_trace_time[self.tb_count]
        downloadEnd = 0
        size_sum = 0
        next_tb_count = self.tb_count
        next_buffer = self.buffer_list[-1]
        reward = 0
        for data_size in self.network_trace_size[self.tb_count:]:
            if size_sum < segmentSize:
                size_sum = size_sum + data_size
                next_tb_count += 1
            else:
                downloadEnd = self.network_trace_time[next_tb_count]
                break
        
        
        
        #print("===============",downloadStart,downloadEnd)
        if self.segmentcount == 0:
            next_buffer += self.segmentDuration
        else:
            next_buffer = next_buffer + self.segmentDuration - (downloadEnd-downloadStart)

        s_ = np.append(action,next_buffer)
        if next_buffer > 1 and next_buffer < 2 :
            if next_tb_count == len(DlRxPhyStats_tbsize):
                reward = 2
                done = True
            else:
                reward = 0
                done = False
        else:
            if self.segmentcount == 0 and downloadEnd-downloadStart < 3 and downloadEnd-downloadStart > 0.8:
                reward = 0
                done = False
            else:
                reward = -1
                done = True

        self.s = s_
        self.tb_count = next_tb_count


        return s_, reward, done

    def record(self):
        # self.bitrate_record.append()
        pass

if __name__ == "__main__":
    env = Environment()
    action = 0
    for i in range(10):
        s_, reward, done = env.step(action)
        print(s_, reward, done)
    