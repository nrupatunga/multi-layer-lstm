import pdb
import numpy as np
from network import LSTM_network


class LSTM:

    def __init__(self, input_txt_file):
        ''' Process the input data, prepare for training
        @params:
        --------
            input_txt_file - text file input containing the test characters
            num_hidden_units - number of hidden units, default=100
            num_layers - number of LSTM layers, default=1
            lr - learning rate, default=0.1
        '''

        pdb.set_trace()
        data = []
        with open(input_txt_file, 'r') as f:
            data = f.read()

        chars = list(set(data))  # list of unique characters
        vocab_size = len(chars)

        #  Characters to index and vice versa
        self.char_to_index = {c: i for i, c in enumerate(chars)}
        self.index_to_char = {i: c for i, c in enumerate(chars)}

        self.vocab_size = vocab_size
        self.data = data
        self.data_len = len(data)

    def build_network(self, seq_len=25, num_hidden_units=100, num_layers=1):
        self.objLstmNet = LSTM_network(self.vocab_size, self.vocab_size, seq_len, num_mem_cells=num_hidden_units, num_layers=num_layers)

    def train(self, learning_rate=0.1):
        ''' Training the LSTM'''

        ptr, sample_n = 0, 0
        objLstmNet = self.objLstmNet

        while True:
            if ptr + self.seq_len >= self.data_len or sample_n is 0:
                h_prev, s_prev = None, None
                ptr = 0

            inputs = [self.char_to_index[ch] for ch in self.data[ptr: ptr + self.seq_len]]
            targets = [self.char_to_index[ch] for ch in self.data[ptr + 1: ptr + 1 + self.seq_len]]
            x_list, y_list = inputs, targets
            num_samples = len(inputs)

            for i in range(num_samples):
                x_one_hot = np.zeros((self.vocab_size))
                x_one_hot[x_list[i]] = 1
                objLstmNet.feed_forward(x_one_hot, s_prev, h_prev)

            s_prev = objLstmNet.lstm_node_list[num_samples - 1].state.s
            h_prev = objLstmNet.lstm_node_list[num_samples - 1].state.h

            objLstmNet.calculate_loss(y_list)
            objLstmNet.feed_backward(y_list)
            objLstmNet.apply_grad()
            objLstmNet.reset()

            if sample_n % 100 == 0:
                text = self.sample_text(inputs[0], h_prev, s_prev, 50)
                print('--------Iter:{} -> Loss: {}--------------'.format(sample_n, self.objLstmNet.loss))
                print(text)
                print('--------------------------------------------------')

            ptr = ptr + self.seq_len
            sample_n = sample_n + 1


if '__main__' == __name__:
    objLSTM = LSTM('./input.txt')
    objLSTM.build_network(seq_len=25, num_hidden_units=100, num_layers=3)
    objLSTM.train(learning_rate=0.1)
