import numpy as np


class LSTM_state:
    ''' Class holding all the states of LSTM unit '''

    def __init__(self, input_dim, output_dim, num_mem_cells):
        '''At time t=0 initialize the states to zeros'''

        self.xc = np.zeros(input_dim)
        self.g = np.zeros(num_mem_cells)
        self.i = np.zeros(num_mem_cells)
        self.f = np.zeros(num_mem_cells)
        self.o = np.zeros(num_mem_cells)
        self.s = np.zeros(num_mem_cells)
        self.h = np.zeros(num_mem_cells)
        self.y = np.zeros(output_dim)
        self.prob = np.zeros(output_dim)
