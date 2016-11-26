import numpy as np


class LSTM_node:
    ''' class LSTM_nodes holds the lstm state, parameters'''

    def __init__(self, lstm_param, lstm_state):
        # store reference to parameters and to activations
        self.state = lstm_state
        self.param = lstm_param
        # non-recurrent input to node
        self.x = None
        # non-recurrent input concatenated with recurrent input
        self.xc = None
        self.y = None

    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def forward_pass(self, x, s_prev=None, h_prev=None):
        ''' LSTM forward pass'''

        # At time t = 0
        if s_prev is None:
            s_prev = np.zeros_like(self.state.s)
        if h_prev is None:
            h_prev = np.zeros_like(self.state.h)

        # save the state
        self.s_prev = s_prev
        self.h_prev = h_prev

        # input concantenation
        xc = np.hstack((x, h_prev))
        self.state.x = x
        self.state.xc = xc
        self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.i = self.sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
        self.state.f = self.sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
        self.state.o = self.sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        self.state.h = np.tanh(self.state.s) * self.state.o
        # self.state.h = self.state.s * self.state.o
        self.state.y = np.dot(self.param.wy, self.state.h) + self.param.by
        pred = self.state.y
        self.state.prob = np.exp(pred) / np.sum(np.exp(pred))
