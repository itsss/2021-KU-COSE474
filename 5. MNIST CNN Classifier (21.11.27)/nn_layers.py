import numpy as np
from skimage.util.shape import view_as_windows

##########
#   convolutional layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_convolutional_layer:

    def __init__(self, Wx_size, Wy_size, input_size, in_ch_size, out_ch_size, std=1e0):
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * Wx_size * Wy_size / 2), (out_ch_size, in_ch_size, Wx_size, Wy_size))
        self.b = 0.01 + np.zeros((1, out_ch_size, 1, 1))
        self.input_size = input_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        wdw = view_as_windows(x, (1,) + self.W.shape[1:])
        y = wdw.reshape(wdw.shape[:4] + (-1,))
        result = y.dot(self.W.reshape(self.W.shape[0], -1).T).transpose(0, 4, 2, 3, 1).squeeze()
        return result.squeeze() + self.b

    def backprop(self, x, dLdy):
        wdw = view_as_windows(x, (1,) + self.W.shape[1:])
        a, b, c, d, e, f, g, h = wdw.shape
        wdw = wdw.reshape(-1, f, g, h)
        d = dLdy.transpose(1, 0, 2, 3)
        d = d.reshape(d.shape[0], -1)
        dLdW = (d.dot(wdw.reshape(wdw.shape[0], -1))).reshape(self.W.shape)

        W = (np.flip(self.W, (2, 3))).transpose(1, 0, 2, 3)
        y = np.pad(dLdy, ((0, 0), (0, 0), (self.W.shape[2] - 1, self.W.shape[2] - 1), (self.W.shape[3] - 1, self.W.shape[3] - 1)), 'constant', constant_values=0)
        wdw = view_as_windows(y, (1,) + W.shape[1:])
        y = wdw.reshape(wdw.shape[:4] + (-1,))

        dLdx = (y.dot(W.reshape((W.shape[0], -1)).T)).transpose(0, 4, 2, 3, 1).squeeze()
        dLdb = np.sum(dLdy, axis=(0, 2, 3), keepdims=True)

        return dLdx, dLdW, dLdb


##########
#   max pooling layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size

    def forward(self, x):
        y = view_as_windows(x, (x.shape[0], x.shape[1], self.pool_size, self.pool_size), step=self.stride)
        y_1 = y.reshape((int(x.shape[2] / self.pool_size), int(x.shape[2] / self.pool_size), x.shape[0], x.shape[1], -1))
        self.y_2 = y_1.transpose(2, 3, 0, 1, 4).reshape(-1, self.pool_size * self.pool_size)
        self.maxarg = np.argmax(self.y_2, axis=1)
        return np.array(np.max(y_1.transpose(2, 3, 0, 1, 4), axis=4))

    def backprop(self, x, dLdy):
        dLdy_ = dLdy.repeat(self.pool_size,axis=2).repeat(self.pool_size,axis=3)
        dydx = np.zeros(self.y_2.shape)
        dydx[np.arange(self.maxarg.shape[0]), self.maxarg] = 1
        return dLdy_ * dydx.reshape(x.shape)

class nn_fc_layer:

    def __init__(self, input_size, output_size, std=1):
        # Xavier/He init
        self.W = np.random.normal(0, std/np.sqrt(input_size/2), (output_size, input_size))
        self.b=0.01+np.zeros((output_size))

    def forward(self,x):
        N = x.shape[0]
        reshaped_x = x.reshape(N, -1)
        out = np.dot(reshaped_x, self.W.T) + self.b.T
        return out

    def backprop(self,x,dLdy):
        reshaped_x = np.reshape(x, (x.shape[0], -1))
        dx = np.reshape(dLdy.dot(self.W), x.shape)
        dw = (reshaped_x.T).dot(dLdy)
        db = np.sum(dLdy, axis=0)
        return dx, dw.T, db

    def update_weights(self,dLdW,dLdb):

        # parameter update
        self.W=self.W+dLdW
        self.b=self.b+dLdb

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b


##########
#   activation layer
##########
#   This is ReLU activation layer.
##########

class nn_activation_layer:

    # performs ReLU activation
    def __init__(self):
        pass

    def forward(self, x):
        return np.maximum(0, x)

    def backprop(self, x, dLdy):
        return dLdy * np.where(x > 0, 1, 0)


##########
#   softmax layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_softmax_layer:

    def __init__(self):
        pass

    def forward(self, x):
        sum_exp_x = np.array([np.sum(np.exp(x), axis=1)]).T
        sum_exp_x = np.repeat(sum_exp_x, x.shape[1], axis=1)
        out = np.exp(x) / sum_exp_x
        return out

    def backprop(self, x, dLdy):
        exp_x = np.exp(x)
        sum_exp_x = np.array([np.sum(exp_x, axis=1)]).T
        sum_exp_x = np.repeat(sum_exp_x, x.shape[1], axis=1)
        sm = exp_x / sum_exp_x

        diag_matrix = sm.reshape(x.shape[0], -1, 1) * np.diag(np.ones(x.shape[1]))
        outer_product = np.matmul(sm.reshape(x.shape[0], -1, 1), sm.reshape(x.shape[0], 1, -1))
        dydx = diag_matrix - outer_product

        dLdx = []
        for batch in range(np.shape(x)[0]):
            dLdy_batch = dLdy[batch]
            dydx_batch = dydx[batch]
            dldy_dydx_dot = dLdy_batch.dot(dydx_batch)
            dLdx.append(dldy_dydx_dot)

        return np.array(dLdx)


##########
#   cross entropy layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_cross_entropy_layer:

    def __init__(self):
        pass

    def forward(self, x, y):
        return -np.sum(np.log(x[np.arange(x.shape[0]), y])) / x.shape[0]

    def backprop(self, x, y):
        dLdx = np.zeros(x.shape)
        dLdx[np.arange(np.shape(x)[0]), y] = -1/x[np.arange(np.shape(x)[0]), y]
        return dLdx / np.shape(x)[0]