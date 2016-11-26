import numpy as np


class LSTM_param:
    ''' Class holding all the LSTM learnable weights and biases of LSTM unit'''

    def random_array(self, mu, sigma, *shape_args):
        return np.random.randn(*shape_args) * sigma + mu

    def init_param(self, num_mem_cells, output_dim, concat_dim, mu=0, sigma=0.01):
        '''initialize the weights'''

        # Weight initialization
        self.wg = self.random_array(mu, sigma, num_mem_cells, concat_dim)
        self.wi = self.random_array(mu, sigma, num_mem_cells, concat_dim)
        self.wf = self.random_array(mu, sigma, num_mem_cells, concat_dim)
        self.wo = self.random_array(mu, sigma, num_mem_cells, concat_dim)
        self.wy = self.random_array(mu, sigma, output_dim, num_mem_cells)

        # Bias initialization
        self.bg = np.zeros(num_mem_cells)
        self.bi = np.zeros(num_mem_cells)
        self.bf = np.zeros(num_mem_cells)
        self.bo = np.zeros(num_mem_cells)
        self.by = np.zeros(output_dim)

        # weight gradient initialization
        self.dwg = np.zeros_like(self.wg)
        self.dwi = np.zeros_like(self.wi)
        self.dwf = np.zeros_like(self.wf)
        self.dwo = np.zeros_like(self.wo)
        self.dwy = np.zeros_like(self.wy)

        # bias gradient initialization
        self.dbg = np.zeros_like(self.bg)
        self.dbi = np.zeros_like(self.bi)
        self.dbf = np.zeros_like(self.bf)
        self.dbo = np.zeros_like(self.bo)
        self.dby = np.zeros_like(self.by)

        # memory variables for adagrad
        self.mwg = np.zeros_like(self.wg)
        self.mwi = np.zeros_like(self.wi)
        self.mwf = np.zeros_like(self.wf)
        self.mwo = np.zeros_like(self.wo)
        self.mwy = np.zeros_like(self.wy)

        self.mbg = np.zeros_like(self.bg)
        self.mbi = np.zeros_like(self.bi)
        self.mbf = np.zeros_like(self.bf)
        self.mbo = np.zeros_like(self.bo)
        self.mby = np.zeros_like(self.by)

    def apply_grad(self, lr=0.1):
        dwg = self.dwg
        dwi = self.dwi
        dwf = self.dwf
        dwo = self.dwo
        dwy = self.dwy

        dbg = self.dbg
        dbi = self.dbi
        dbf = self.dbf
        dbo = self.dbo
        dby = self.dby

        self.mwg += dwg * dwg
        self.mwi += dwi * dwi
        self.mwf += dwf * dwf
        self.mwo += dwo * dwo
        self.mwy += dwy * dwy
        self.wg += -lr * dwg / np.sqrt(self.mwg + 1e-8)
        self.wi += -lr * dwi / np.sqrt(self.mwi + 1e-8)
        self.wf += -lr * dwf / np.sqrt(self.mwf + 1e-8)
        self.wo += -lr * dwo / np.sqrt(self.mwo + 1e-8)
        self.wy += -lr * dwy / np.sqrt(self.mwy + 1e-8)

        self.mbg += dbg * dbg
        self.mbi += dbi * dbi
        self.mbf += dbf * dbf
        self.mbo += dbo * dbo
        self.mby += dby * dby
        self.bg += -lr * dbg / np.sqrt(self.mbg + 1e-8)
        self.bi += -lr * dbi / np.sqrt(self.mbi + 1e-8)
        self.bf += -lr * dbf / np.sqrt(self.mbf + 1e-8)
        self.bo += -lr * dbo / np.sqrt(self.mbo + 1e-8)
        self.by += -lr * dby / np.sqrt(self.mby + 1e-8)

    def __init__(self, input_dim, output_dim, num_mem_cells=100):
        ''' Initialize weights, bias,  character indexing
        @params:
        --------
            input_dim - length of the input sequence to the lstm
            num_mem_cells(optional)  - Number of memory cells, default num_mem_cells = 100
            lr(optional) - learning rate, default lr = 0.1
        '''

        self.num_mem_cells = num_mem_cells
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.concat_dim = input_dim + num_mem_cells

        # Init parameters
        self.init_param(num_mem_cells, output_dim, self.concat_dim)
