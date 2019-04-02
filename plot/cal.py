import os
import os.path
import math
import fileinput
import matplotlib.pyplot as plt
import random

playback_file = "sim4_cl0_playbackLog.txt"
through_file = "tj_phy4-10_cl0.txt"
download_file = "sim4_cl0_downloadLog.txt"

data_repindex,data_timea = [0],[0]
data_th,data_timet = [],[]
data_dl,data_timed = [0],[0]
segmentDuration = 10
timetemp = 0
x_temp = 0
#--------------------------------------------------
with open(playback_file, 'r') as RB_to_read:
    n=0
    while True:
        lines = RB_to_read.readline() # 整行读取数据
        if not lines:
            break
        if n == 0:
            i = lines.split()
            data_repindex.append(0)
            x_temp = float(i[1])
            data_timea.append(x_temp)
            data_repindex.append((float(i[2])+1)*1e7)
            data_timea.append(x_temp)
            data_repindex.append((float(i[2])+1)*1e7)
            data_timea.append(x_temp+segmentDuration)
            timetemp = x_temp+segmentDuration
        else:    
            i = lines.split()
            x_temp = float(i[1])
            if abs(timetemp-x_temp)>0.000001:
                data_repindex.append(0)
                data_timea.append(timetemp)
                data_repindex.append(0)
                data_timea.append(x_temp)
            #print(timetemp,x_temp)
            data_repindex.append((float(i[2])+1)*1e7)
            data_timea.append(x_temp)
            data_repindex.append((float(i[2])+1)*1e7)
            data_timea.append(x_temp+segmentDuration)
            timetemp = x_temp+segmentDuration
        n = n+1

#print(data_timea,data_repindex)
#--------------------------------------------------
with open(through_file,'r') as th_to_read:
    while True:
        lines = th_to_read.readline()
        if not lines:
            break
        i = lines.split()
        data_timet.append(float(i[0]))
        data_th.append(float(i[1]))

#--------------------------------------------------
timed_temp = 0
with open(download_file,'r') as dl_to_read:
    n = 0
    while True:
        lines = dl_to_read.readline()
        if not lines:
            break
        i = lines.split()
        if n==0:
            data_timed.append(float(i[1]))
            data_dl.append(0)
            data_timed.append(float(i[1]))
            data_dl.append(float(i[5]))
            data_timed.append(float(i[3]))
            data_dl.append(float(i[5]))
            timed_temp = float(i[3])
        else:
            #print(timed_temp,float(i[1]))
            if (float(i[1])-timed_temp)>0.0001:
                #print("this is here!!")
                data_timed.append(timed_temp)
                data_dl.append(0)
                data_timed.append(float(i[1]))
                data_dl.append(0)
            data_timed.append(float(i[1]))
            data_dl.append(float(i[5]))
            data_timed.append(float(i[3]))
            data_dl.append(float(i[5]))
            timed_temp = float(i[3])
        n = n+1
#print(data_timea,data_repindex)

tx_ave,tx_data = [],[]
tx_tbsize = 0 
with open(download_file,'r') as dl_to_read:
    n = 0
    while True:
        lines = dl_to_read.readline()
        if not lines:
            break
        i = lines.split()
        for x,y in zip(data_timet,data_th):
            if x>float(i[1]) and x<float(i[3]):
                tx_tbsize = tx_tbsize + y
                n = n+1
            elif x>=float(i[3]):
                tx_ave.append(tx_tbsize/n)
                tx_data.append(float(i[4]))
                tx_tbsize = 0
                n = 0
                break
#print(tx_ave,tx_data)
tx_pl = []
with open(playback_file,'r') as pl_to_read:
    n = 0
    while True:
        lines = pl_to_read.readline()
        if not lines:
            break
        i = lines.split()
        tx_pl.append((float(i[2])+1)*10000000)

print(tx_pl)
data_tx = []
for x,y in zip(tx_ave,tx_data):
    data_tx.append(abs(1-y/x))


n=0
rate_dl,time_dl,rate_pl,time_pl = [],[],[],[]
#print(len(data_timed))
for x,y in zip(data_timet,data_th):
    #print(n)
    if x<92 and y!=0 and n<len(data_timed):
        if x>data_timed[n] and x<data_timed[n+1]:
            if abs(data_dl[n]-y)/y<10:
                rate_dl.append(abs(data_dl[n]-y)/y)
                time_dl.append(x)
        elif x>=data_timed[n+1]:
            n = n+2

n=0
for x,y in zip(data_timet,data_th):
    if x<92 and y!=0 and n<len(data_timed):
        if x>data_timea[n] and x<data_timea[n+1]:
            if abs(data_repindex[n]-y)/y<10:
                rate_pl.append(abs(data_repindex[n]-y)/y)
                time_pl.append(x)
        elif x>=data_timed[n+1]:
            n = n+2

ave_bl = sum(rate_dl)/len(rate_dl)
ave_pl = sum(rate_pl)/len(rate_pl)
ave_tx = sum(data_tx)/len(data_tx)
print("average dl/th_rate: ",ave_bl)
print("average pl/th_rate: ",ave_pl)
print("average tx/th_rate: ",ave_tx)

choose = 4
plt.figure(figsize=(80,60))
if choose==4:
    c = plt.subplot(111)
    c1 = c.plot(data_timea,data_repindex,'b-',label = 'RepBitrate',linewidth=4.0)
    c2 = c.plot(data_timet,data_th,'r-',label = 'BW',linewidth=4.0)
    c3 = c.plot(data_timed,data_dl,'g-',label = 'DownloadRate',linewidth=4.0)
else:
    c = plt.subplot(111)
    c1 = c.plot(time_dl,rate_dl,'r-',label = 'dl/th_rate',linewidth=4.0)
    c2 = c.plot(time_pl,rate_pl,'b-',label = 'pl/th_rate',linewidth=4.0)
    
plt.grid(True)

if choose==4:
    plt.xlabel("Time/s",fontsize=100)
    plt.ylabel("Bitrate/100Mbps",fontsize=100)
    plt.xticks(fontsize=80)
    plt.yticks(fontsize=80)
    plt.ylim(0,150000000)
    plt.xlim(0,90)
    handlesa,labelsa = c.get_legend_handles_labels()
    c.legend(handlesa[::-1],labelsa[::-1],fontsize=100)
    plt.savefig("cal4-10.jpg")
else:
    plt.xlabel("Time/s",fontsize=100)
    plt.ylabel("Rate",fontsize=100)
    plt.xticks(fontsize=80)
    plt.yticks(fontsize=80)
    plt.ylim(0,10)
    plt.xlim(0,90)
    handlesa,labelsa = c.get_legend_handles_labels()
    c.legend(handlesa[::-1],labelsa[::-1],fontsize=100)
    plt.savefig("rate4-10.jpg")




f = open('sim4_cl0_downloadLog.txt')
byte = 0
n = 0
while 1:
    line = f.readline();
    if not line:
        break;
    if float(line.split()[1])<92:
        byte =  byte + int(line.split()[4])
        n = n+1
f.close()
print(byte*0.8/n)
temp1 = byte*0.8/n

f = open('tj_phy4-10_cl0.txt')
#f = open('sim4_cl0_downloadLog.txt')
byte = 0
n = 0
while 1:
    line = f.readline();
    if not line:
        break;
    if float(line.split()[0])<92:
        byte =  byte + int(line.split()[1])
        n = n+1
    #byte =  byte + int(line.split()[4])
f.close()
print(byte/n)
temp2=byte/n
print(1-temp1/temp2)
temp3 = sum(tx_pl)/len(tx_pl)
temp4 = byte/n
print("plallrate: ",n,1-temp3/temp4)
