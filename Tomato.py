import numpy as np 

n_actions = 10

def Tomato(throughput, last_bitrate, buffer):
    if throughput < 1e7:
        bitrate = 0
    elif throughput < 2e7:
        bitrate = 1
    elif throughput < 3e7:
        bitrate = 2
    elif throughput < 4e7:
        bitrate = 3
    elif throughput < 5e7:
        bitrate = 4
    elif throughput < 6e7:
        bitrate = 5
    elif throughput < 7e7:
        bitrate = 6
    elif throughput < 8e7:
        bitrate = 7
    elif throughput < 9e7:
        bitrate = 8
    else:
        bitrate = 9

    if buffer < 0.5:
        bitrate -= 2
    elif buffer < 1.5:
        bitrate -= 1
    elif buffer >= 2.5:
        bitrate += 1
    
    if bitrate < 0:
        bitrate = 0
    elif bitrate > n_actions-1:
        bitrate = n_actions
    

    return bitrate


if __name__ == "__main__":
    throughput = 54000000
    last_bitrate = 1
    buffer = 3.4
    bitrate = Tomato(throughput, last_bitrate, buffer)
    print(bitrate)

    