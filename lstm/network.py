import numpy as np
from param import LSTM_param
from state import LSTM_state
from node import LSTM_node


class LSTM_network:

    def __init__(self, input_dim, output_dim, seq_len, num_mem_cells=100, num_layers=1):
        '''Initialize the LSTM unit, LSTM state'''

        self.lstm_param = {}
        self.lstm_node_list = {}
        self.seq_len = seq_len
        self.loss = 0
        self.num_mem_cells = num_mem_cells
        self.num_layers = num_layers

        for i in range(num_layers):
            # weights and bias are reused, so initialize lstm_param only once
            self.lstm_param[i] = LSTM_param(input_dim, output_dim, num_mem_cells)

        for i in range(num_layers):
            self.lstm_node_list[i] = {}
            for j in range(seq_len):
                lstm_state = LSTM_state(input_dim, output_dim, num_mem_cells)
                self.lstm_node_list[i][j] = LSTM_node(self.lstm_param, lstm_state)

        # loss
        self.smooth_loss = -np.log(1.0 / output_dim) * self.seq_len  # loss at iteration 0

    def reset(self):
        self.lstm_param = {}
        self.lstm_node_list = {}
