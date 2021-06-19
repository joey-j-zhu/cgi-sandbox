# Stolen from an EECS 126 lab
import numpy as np

def smooth(t):
    return t*t*t*(t*(t*6 - 15) + 10)

class SmoothStream:
    def __init__(self, start_frame, buf_size, rate):
        self.out = start_frame
        self.total = np.zeros(start_frame.shape)
        self.stream = []
        self.buf_size = buf_size
        self.rate = rate
        self.size = 0

    def add_frame(self, frame):
        self.stream += [frame + np.zeros(frame.shape)]
        self.total += frame
        #print(self.total)
        if self.size == self.buf_size:
            #print(self.stream[0] - frame)
            self.total -= self.stream.pop(0)
        else:
            self.size += 1

    # Current raster is attracted to a moving average of previous frames
    def step(self):
        self.out += (self.total / self.size - self.out) * self.rate
