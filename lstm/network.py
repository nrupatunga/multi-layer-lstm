import numpy as np
from param import LSTM_param
from state import LSTM_state
from node import LSTM_node


class LSTM_network:

    def __init__(self, input_dim, output_dim, seq_len, num_mem_cells=100, num_layers=1):
        '''Initialize the LSTM unit, LSTM state
        @params:
        --------
            input_dim - dimension of the input
            output_dim - dimension of the output
            seq_len - sequence lenght / batch size
        '''

        self.lstm_param = {}
        self.lstm_node_list = {}
        self.seq_len = seq_len
        self.loss = 0
        self.num_mem_cells = num_mem_cells
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        if num_layers is 1:
            self.lstm_param[0] = LSTM_param(self.input_dim, self.output_dim, self.num_mem_cells)
            self.lstm_node_list[0] = {}
            for j in range(self.seq_len):
                lstm_state = LSTM_state(self.input_dim, self.output_dim, self.num_mem_cells)
                self.lstm_node_list[0][j] = LSTM_node(self.lstm_param[0], lstm_state)
        else:
            # Input layer
            self.lstm_param[0] = LSTM_param(self.input_dim, self.num_mem_cells, self.num_mem_cells)
            self.lstm_node_list[0] = {}
            for j in range(self.seq_len):
                lstm_state = LSTM_state(self.input_dim, self.num_mem_cells, self.num_mem_cells)
                self.lstm_node_list[0][j] = LSTM_node(self.lstm_param[0], lstm_state)

            # Hidden layers
            for i in range(1, self.num_layers - 1):
                self.lstm_node_list[i] = {}
                self.lstm_param[i] = LSTM_param(self.num_mem_cells, self.num_mem_cells, self.num_mem_cells)
                for j in range(self.seq_len):
                    lstm_state = LSTM_state(self.num_mem_cells, self.num_mem_cells, self.num_mem_cells)
                    self.lstm_node_list[i][j] = LSTM_node(self.lstm_param[i], lstm_state)

            # Output layer
            self.lstm_param[self.num_layers - 1] = LSTM_param(self.num_mem_cells, self.output_dim, self.num_mem_cells)
            self.lstm_node_list[self.num_layers - 1] = {}
            for j in range(self.seq_len):
                lstm_state = LSTM_state(self.num_mem_cells, self.output_dim, self.num_mem_cells)
                self.lstm_node_list[self.num_layers - 1][j] = LSTM_node(self.lstm_param[self.num_layers - 1], lstm_state)

        # loss
        self.smooth_loss = -np.log(1.0 / output_dim) * self.seq_len  # loss at iteration 0

    def reset(self):
        ''' Reset all the LSTM layer before next batch'''

        self.lstm_node_list[0] = {}
        for j in range(self.seq_len):
            lstm_state = LSTM_state(self.input_dim, self.num_mem_cells, self.num_mem_cells)
            self.lstm_node_list[0][j] = LSTM_node(self.lstm_param[0], lstm_state)

        for layer in range(1, self.num_layers - 1):
            self.lstm_node_list[layer] = {}
            for j in range(self.seq_len):
                lstm_state = LSTM_state(self.num_mem_cells, self.num_mem_cells, self.num_mem_cells)
                self.lstm_node_list[layer][j] = LSTM_node(self.lstm_param[layer], lstm_state)

        self.lstm_node_list[self.num_layers - 1] = {}
        for j in range(self.seq_len):
            lstm_state = LSTM_state(self.num_mem_cells, self.output_dim, self.num_mem_cells)
            self.lstm_node_list[self.num_layers - 1][j] = LSTM_node(self.lstm_param[self.num_layers - 1], lstm_state)

    def calculate_loss(self, target, num_layers):
        ''' Cross entropy loss'''

        assert len(self.lstm_node_list[num_layers - 1]) == len(target)

        loss = 0
        for i, tt in enumerate(target):
            prob = self.lstm_node_list[num_layers - 1][i].state.prob
            loss += -np.log(prob[tt])

        self.loss = loss

    def feed_forward(self, input_x, t_idx, num_layers, s_prev=None, h_prev=None):
        '''Storing input sequence, add new state everytime there is a new input
        @params:
        --------
            input_x - input vector to LSTM, x(t)
            t_idx - time index
            num_layers - number of layers in the network
            s_prev - previous cell state
            h_prev - previous hidden state
        '''

        # Forward pass
        layer_input, is_output_layer = input_x, False
        for layer in range(num_layers):
            if layer is num_layers - 1:
                is_output_layer = True
            # first iteration at time t = 0
            if t_idx == 0 and not bool(s_prev) and not bool(h_prev):
                self.lstm_node_list[layer][t_idx].forward_pass(layer_input, is_output_layer=is_output_layer)
            # second iteration at time t = 1,... so on
            elif t_idx == 0 and bool(s_prev) and bool(h_prev):
                self.lstm_node_list[layer][t_idx].forward_pass(layer_input, s_prev[layer], h_prev[layer], is_output_layer=is_output_layer)
            else:
                self.s_prev = self.lstm_node_list[layer][t_idx - 1].state.s
                self.h_prev = self.lstm_node_list[layer][t_idx - 1].state.h
                self.lstm_node_list[layer][t_idx].forward_pass(layer_input, self.s_prev, self.h_prev, is_output_layer=is_output_layer)

            layer_input = self.lstm_node_list[layer][t_idx].state.h

    def feed_backward(self, target, num_layers):
        ''' Backpropagation
        @params:
        --------
            target - ground truth labels
            num_layers - number of LSTM layers
        '''

        for layer in range(num_layers):
            assert len(self.lstm_node_list[layer]) == len(target)

        dh_next, ds_next = {}, {}
        idx = 0
        # for each batch, we need to initialize with zero
        for layer in range(num_layers):
            # weight gradient initialization
            self.lstm_node_list[layer][idx].param.dwg = np.zeros_like(self.lstm_node_list[layer][idx].param.wg)
            self.lstm_node_list[layer][idx].param.dwi = np.zeros_like(self.lstm_node_list[layer][idx].param.wi)
            self.lstm_node_list[layer][idx].param.dwf = np.zeros_like(self.lstm_node_list[layer][idx].param.wf)
            self.lstm_node_list[layer][idx].param.dwo = np.zeros_like(self.lstm_node_list[layer][idx].param.wo)
            self.lstm_node_list[layer][idx].param.dwy = np.zeros_like(self.lstm_node_list[layer][idx].param.wy)

            # bias gradient initialization
            self.lstm_node_list[layer][idx].param.dbi = np.zeros_like(self.lstm_node_list[layer][idx].param.bi)
            self.lstm_node_list[layer][idx].param.dbf = np.zeros_like(self.lstm_node_list[layer][idx].param.bf)
            self.lstm_node_list[layer][idx].param.dbo = np.zeros_like(self.lstm_node_list[layer][idx].param.bo)
            self.lstm_node_list[layer][idx].param.dby = np.zeros_like(self.lstm_node_list[layer][idx].param.by)

            # gradient w.r.t h(t-1) and s(t-1)
            dh_next[layer] = np.zeros_like(self.lstm_node_list[layer][0].state.h)
            ds_next[layer] = np.zeros_like(self.lstm_node_list[layer][0].state.s)

        for i, tt in reversed(list(enumerate(target))):
            for layer in reversed(range(num_layers)):
                # print('Param Address: {}, State Address: {}'.format(id(self.lstm_node_list[i].param), id(self.lstm_node_list[i].state)))
                param = self.lstm_node_list[layer][i].param
                state = self.lstm_node_list[layer][i].state
                x_dim = param.input_dim
                xc = self.lstm_node_list[layer][i].state.xc

                if layer == num_layers - 1:
                    dy = np.copy(self.lstm_node_list[layer][i].state.prob)
                    dy[tt] -= 1
                    self.lstm_node_list[layer][idx].param.dwy += np.outer(dy, state.h)
                    self.lstm_node_list[layer][idx].param.dby += dy
                    dh = np.dot(param.wy.T, dy)

                dh += dh_next[layer]
                # ds = dh * state.o + ds_next[layer]
                ds = dh * state.o * (1 - (np.tanh(state.s) ** 2)) + ds_next[layer]

                # gradients till the non linearities for gates
                dg = state.i * ds
                di = state.g * ds
                df = self.lstm_node_list[layer][i].s_prev * ds
                do = dh * np.tanh(state.s)

                # gradients including non-linearities
                dg_input = (1.0 - state.g ** 2) * dg
                di_input = (1.0 - state.i) * state.i * di
                df_input = (1.0 - state.f) * state.f * df
                do_input = (1.0 - state.o) * state.o * do

                # Update gradients
                self.lstm_node_list[layer][idx].param.dwg += np.outer(dg_input, xc)
                self.lstm_node_list[layer][idx].param.dwi += np.outer(di_input, xc)
                self.lstm_node_list[layer][idx].param.dwf += np.outer(df_input, xc)
                self.lstm_node_list[layer][idx].param.dwo += np.outer(do_input, xc)
                self.lstm_node_list[layer][idx].param.dbg += dg_input
                self.lstm_node_list[layer][idx].param.dbi += di_input
                self.lstm_node_list[layer][idx].param.dbf += df_input
                self.lstm_node_list[layer][idx].param.dbo += do_input

                ds_next[layer] = state.f * ds
                # compute bottom diff
                dxc = np.zeros_like(xc)
                dxc += np.dot(param.wi.T, di_input)
                dxc += np.dot(param.wf.T, df_input)
                dxc += np.dot(param.wo.T, do_input)
                dxc += np.dot(param.wg.T, dg_input)
                dh_next[layer] = dxc[x_dim:]
                dh = dxc[:x_dim]

    def apply_grad(self, lr):
        # update your weights
        self.smooth_loss = self.smooth_loss * 0.999 + self.loss * 0.001
        for layer in range(self.num_layers):
            self.lstm_param[layer].apply_grad(lr)
