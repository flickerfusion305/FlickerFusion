import numpy as np
import json
class DecayThenFlatSchedule():

    def __init__(self,
                 start,
                 finish,
                 time_length,
                 start_time,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay
        self.start_time=start_time


        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * abs(T-self.start_time))
        
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- abs(T-self.start_time) / self.exp_scaling)))
    pass