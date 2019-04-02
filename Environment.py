import numpy as np


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
        self.segmentsize_list = []
        self.segment_dltime_list = []
        self.buffer_list = [0]
        self.th_endtime = []
        self.rebuffer_starttime_list = []
        self.rebuffer_endtime_lsit = []
        self.tb_count = 0
        self.s = np.array([0.,0.,0.])

        # plot
        self.plot_buffer_time = [0]
        self.plot_buffer_data = [0]

    def reset(self):
        origin = np.array([0.,0.,0.]) # [throughput, bitrate, buffer]

        return origin


    def step(self, action):
        # s = 0
        s_ = self.s
        reward = 0
        done = False

        action = int(action)
        if self.segmentcount < len(self.video_trace[action]):
            segmentSize = self.video_trace[action][self.segmentcount]*self.segmentDuration*8
        else:
            segmentSize = 0
        downloadStart = self.network_trace_time[self.tb_count]
        downloadEnd = downloadStart
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
        if self.segmentcount == 0:
            next_buffer += self.segmentDuration
            # plot
            if downloadEnd == downloadStart:
                pass
            else:
                self.plot_buffer_time.append(downloadStart)
                self.plot_buffer_data.append(0)
                self.plot_buffer_time.append(downloadEnd)
                self.plot_buffer_data.append(0)
                self.plot_buffer_time.append(downloadEnd)
                self.plot_buffer_data.append(next_buffer)
            # print("self.segmentcount == 0!!")
        else: 
            if next_buffer <= (downloadEnd-downloadStart):
                self.rebuffer_starttime_list.append(downloadStart + next_buffer)
                self.rebuffer_endtime_lsit.append(downloadEnd)
                
                # plot 
                if downloadEnd == downloadStart:
                    pass
                else:
                    self.plot_buffer_time.append(downloadStart + next_buffer)
                    self.plot_buffer_data.append(0)
                    self.plot_buffer_time.append(downloadEnd)
                    self.plot_buffer_data.append(0)
                    self.plot_buffer_time.append(downloadEnd)
                    self.plot_buffer_data.append(self.segmentDuration)
                next_buffer = self.segmentDuration
                # print("rebuff !!")
            else:
                if downloadEnd == downloadStart:
                    pass
                else:
                    self.plot_buffer_time.append(downloadEnd)
                    self.plot_buffer_data.append(next_buffer - (downloadEnd-downloadStart))

                    next_buffer = next_buffer + self.segmentDuration - (downloadEnd-downloadStart)
                    # plot
                    self.plot_buffer_time.append(downloadEnd)
                    self.plot_buffer_data.append(next_buffer)
                # print("not rebuff !!")
        
        

        # print("===============",downloadStart,downloadEnd,self.buffer_list[-1],next_buffer)
        
        # self.record(action,next_buffer,segmentSize,downloadEnd-downloadStart,downloadEnd)

        
        # if next_buffer > 1 :#and next_buffer < 2 :
        #     if next_tb_count == len(DlRxPhyStats_tbsize):
        #         reward = 2
        #         done = True
        #     else:
        #         reward = 0
        #         done = False
        # else:
        #     if self.segmentcount == 0 and downloadEnd-downloadStart < 3 and downloadEnd-downloadStart > 0.8:
        #         reward = 0
        #         done = False
        #     else:
        #         reward = -1
        #         done = True
        if segmentSize == 0 or next_tb_count >= len(self.network_trace_time):
            done = True
        else:
            self.record(action,next_buffer,segmentSize,downloadEnd-downloadStart,downloadEnd)
            
        s_[0] = self.segmentsize_list[-1]/self.segment_dltime_list[-1]
        s_[1] = (action + 1)*1e7
        s_[2] = next_buffer
        self.segmentcount += 1
        self.s = s_
        self.tb_count = next_tb_count


        return s_, reward, done

    def record(self, bitrate, buffer, segmentsize, downloadtime, downloadEnd):
        self.bitrate_record.append((bitrate+1)*1e7)
        self.buffer_list.append(buffer)
        self.segmentsize_list.append(segmentsize)
        self.segment_dltime_list.append(downloadtime)
        self.th_endtime.append(downloadEnd)

    def th_plot(self):
        data_size,data_time = [],[]
        th_size, th_time = [],[]
        interval = 0.5 #10*1000
        segmentstart = 0
        timetemp = 0
        #--------------------------------------------------
        with open("sim0_cl0_throughputLog.txt", 'r') as phyrate_to_read:
            n=0
            while True:
                lines = phyrate_to_read.readline() 
                if not lines:
                    break   
                i = lines.split()
                temp_time = float(i[0])
                # print(i,temp_time,segmentstart,data_time)
                if temp_time <= segmentstart + interval:
                    data_size.append(float(i[1]))
                    data_time.append(float(i[0]))
                else:
                    segmentstart = temp_time
                    if len(data_size) == 0 or (data_time[-1]-data_time[0]) == 0:
                        th_size.append(0)
                        th_time.append(temp_time)
                    else:
                        th_size.append(sum(data_size)*8/(data_time[-1]-data_time[0]))
                        th_time.append(sum(data_time)/len(data_time))
                    data_size, data_time = [], []
                    
                n = n+1


        plot_bitrate, plot_bitrate_time = [0,0], [0,self.th_endtime[0] - self.segment_dltime_list[0]]
        th = [i/j for i,j in zip(self.segmentsize_list, self.segment_dltime_list)]
        for i in range(len(self.bitrate_record)):
            plot_bitrate_time.append(self.th_endtime[i] - self.segment_dltime_list[i])
            plot_bitrate_time.append(self.th_endtime[i])
            plot_bitrate.append(self.bitrate_record[i])
            plot_bitrate.append(self.bitrate_record[i])
        import matplotlib.pyplot as plt
        plt.figure(figsize=(40,24))
        c = plt.subplot(111)
        c1 = c.plot(self.th_endtime, th,'r-',label = 'Throughput',linewidth=2.0)
        c2 = c.plot(plot_bitrate_time, plot_bitrate,'g-',label = 'Bitrate',linewidth=2.0)
        c3 = c.plot(th_time,th_size,'b-',label = 'sim-Throughput',linewidth=2.0)
        plt.grid(True)
        plt.xlabel("Time/s",fontsize=20)
        plt.ylabel("Bitrate/100Mbps",fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(0,150000000)
        plt.xlim(0,90)
        handlesa,labelsa = c.get_legend_handles_labels()
        c.legend(handlesa[::-1],labelsa[::-1],fontsize=20)
        plt.savefig("Throughput.png")
        # plt.show()

    def buffer_plot(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(40,24))
        c = plt.subplot(111)
        c1 = c.plot(self.plot_buffer_time[:-1], self.plot_buffer_data[:-1],'r-',label = 'Buffer',linewidth=2.0)
        plt.grid(True)
        plt.xlabel("Time/s",fontsize=20)
        plt.ylabel("Buffer/s",fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(0,15)
        plt.xlim(0,90)
        handlesa,labelsa = c.get_legend_handles_labels()
        c.legend(handlesa[::-1],labelsa[::-1],fontsize=20)
        plt.savefig("buffer.png")
        # plt.show()

if __name__ == "__main__":
    env = Environment()
    action = 6
    s_, reward, done = env.step(action)
    while not done:
        # print(done)
        s_, reward, done = env.step(action)
        # print(s_, reward, done)
    # env.th_plot()
    # env.buffer_plot()
    # print(env.plot_buffer_time)
    # print(env.plot_buffer_data)
    # print(env.segment_dltime_list)