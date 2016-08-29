from chainer import Chain, Variable
import chainer.initializers as init
import chainer.functions as F
from chainer.functions.activation import lstm
import chainer.links as L
import numpy as np

"""
Note: We may want to make use of batch normalization to ensure that hyperparams are robust over problem settings:

Lillicrap, T.P., Hunt, J.J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., Wierstra, D., 2015. Continuous control with deep reinforcement learning.

"""

class MLP(Chain):
    """
    Model implements actor and critic with shared model
    MLP variant; supports multiple frames as input
    """

    def __init__(self, ninput, nhidden, noutput, nframes=1):
        """

        :param ninput:
        :param nhidden:
        :param noutput: number of action outputs
        """
        super(MLP, self).__init__(
            l1=L.Linear(nframes*ninput, nhidden, initialW=init.HeNormal()),
            pi=L.Linear(nhidden, noutput, initialW=init.HeNormal()),
            v=L.Linear(nhidden, 1, initialW=init.HeNormal())
        )

        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput
        self.nframes = nframes

        self.reset()

        # Note that pi and v define the actor and critic, respectively

    def __call__(self, x, persistent=False, internal_states=False):
        """

        :param x: sensory input
        :param persistent: whether or not to retain the internal state (if any)
        :param internal_states: whether or not to return internal states
        :return: policy output pi and value v
        """

        # add stimulus to buffer
        if self.nframes == 1:
            h = F.relu(self.l1(Variable(x)))
        else:

            if persistent:
                _buffer = self.buffer

            if self.idx >= self.nframes - 1:
                self.buffer = np.vstack([self.buffer[1:self.nframes], x])
            else:
                self.buffer[self.idx] = x

            h = F.relu(self.l1(Variable(self.buffer.reshape([1, self.buffer.size]))))

            if persistent:
                self.buffer = _buffer
            else:
                self.idx += 1

        pi = self.pi(h)
        v = self.v(h)

        if internal_states:
            return pi, v, {'hidden state': h.data[0]}
        else:
            return pi, v

    def unchain_backward(self):
        pass


    def reset(self):

        if self.nframes > 1:
            self.buffer = np.zeros([self.nframes, self.ninput], dtype=np.float32)
            self.idx = 0


    def get_internal(self):
        """
        Returns: internal states
        """



class CNN(Chain):
    """
    Convolutional neural network
    """

    def __init__(self, ninput, nhidden, noutput, nframes = 1):
        """

        :param ninput:
        :param nhidden:
        :param noutput: number of action outputs
        """
        super(CNN, self).__init__(
            # dependence between filter size and padding; here output still 20x20 due to padding
            l1=L.Convolution2D(nframes, nhidden, 3, 1, 1, initialW=init.HeNormal()),
            pi=L.Linear(ninput[0] * ninput[1] * nhidden, noutput, initialW=init.HeNormal()),
            v=L.Linear(ninput[0] * ninput[1] * nhidden, 1, initialW=init.HeNormal()),
        )

        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput
        self.nframes = nframes

        self.reset()

    def __call__(self, x, persistent = False, internal_states=False):
        """

        :param x: sensory input (1 x nframes x ninput[0] x ninput[1])
        :param persistent: whether or not to retain the internal state (if any)
        :param internal_states: whether or not to return internal states
        :return: policy output pi and value v
        """

        # add stimulus to buffer
        if self.nframes == 1:
            h = F.relu(self.l1(Variable(x)))
        else:

            if persistent:
                _buffer = self.buffer

            if self.idx >= self.nframes - 1:
                self.buffer = np.vstack([self.buffer[1:self.nframes], x])
            else:
                self.buffer[self.idx] = x

            h = F.relu(self.l1(Variable(np.expand_dims(self.buffer, axis = 0))))

            if persistent:
                self.buffer = _buffer
            else:
                self.idx += 1

        pi = self.pi(h)
        v = self.v(h)

        if internal_states:
            return pi, v, {'hidden state': h.data[0]}
        else:
            return pi, v

    def unchain_backward(self):
        pass

    def reset(self):

        if self.nframes > 1:
            self.buffer = np.zeros([self.nframes, self.ninput[0], self.ninput[1]], dtype=np.float32)
            self.idx = 0


class RNN(Chain):
    """
    Model implements actor and critic with shared model
    """

    def __init__(self, ninput, nhidden, noutput):
        """

        :param ninput:
        :param nhidden:
        :param noutput: number of action outputs
        """
        super(RNN, self).__init__(
            l1=L.LSTM(ninput, nhidden),
            pi=L.Linear(nhidden, noutput, initialW=init.HeNormal()),
            v=L.Linear(nhidden, 1, initialW=init.HeNormal()),
        )

        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput

        # Note that pi and v define the actor and critic, respectively

    def __call__(self, x, persistent=False, internal_states=False):
        """

        :param x: sensory input
        :param persistent: whether or not to retain the internal state (if any)
        :param internal_states: whether or not to return internal states
                h = hidden states
                c = cell states
                a = cell input
                i = input gates
                f = forget gates
                o = output gates

        c &= \\tanh(a) \\text{sigmoid}(i)
           + c_{\\text{prev}} \\text{sigmoid}(f), \\\\
        h &= \\tanh(c) \\text{sigmoid}(o).

        http://www.felixgers.de/papers/phd.pdf
        http://docs.chainer.org/en/stable/_modules/chainer/functions/activation/lstm.html

        :return: policy output pi and value v

        """

        if persistent:
            _c, _h = self.get_persistent()

        h = self.l1(Variable(x))
        pi = self.pi(h)
        v = self.v(h)

        if internal_states:
            c = self.l1.c
            a, i, f, o = lstm._extract_gates(
                self.l1.upward.b.data.reshape(1, 4 * c.data.size, 1))

        if persistent:
            self.set_persistent(_c, _h)

        if internal_states:
            return pi, v, {'hidden state': h.data[0], 'cell state': c.data[0], 'cell input': a.squeeze(),
                           'input gates': i.squeeze(), 'forget gates': f.squeeze(), 'output gates': o.squeeze()}
        else:
            return pi, v

    def reset(self):
        self.l1.reset_state()

    def get_persistent(self):
        return [self.l1.c, self.l1.h]

    def set_persistent(self, _c, _h):
        self.l1.c = _c
        self.l1.h = _h

    def unchain_backward(self):
        if not self.l1.h is None:
            self.l1.h.unchain_backward()
        if not self.l1.c is None:
            self.l1.c.unchain_backward()


class CRNN(Chain):
    """
    Convolutional recurrent neural network
    """

    def __init__(self, ninput, nhidden, noutput):
        super(CRNN, self).__init__(
            l1=L.Convolution2D(1, nhidden, 3, 1, 1), # HeNormal/GlorotNormal gives overflow errors
            l2=L.LSTM(ninput[0] * ninput[1] * nhidden, nhidden),
            pi=L.Linear(nhidden, noutput, initialW=init.HeNormal()),
            v=L.Linear(nhidden, 1, initialW=init.HeNormal()),
        )

        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput

    def __call__(self, x, persistent=False, internal_states=False):
        """

        :param x: sensory input
        :param persistent: whether or not to retain the internal state (if any)
        :param internal_states: whether or not to return internal states
        :return: policy output pi and value v

        """

        if persistent:
            _c, _h = self.get_persistent()

        h1 = F.relu(self.l1(Variable(np.expand_dims(x, axis=0))))
        h2 = self.l2(h1)
        pi = self.pi(h2)
        v = self.v(h2)

        if internal_states:
            c2 = self.l2.c
            a2, i2, f2, o2 = lstm._extract_gates(
                self.l2.upward.b.data.reshape(1, 4 * c2.data.size, 1))

        if persistent:
            self.set_persistent(_c, _h)

        if internal_states:
            return pi, v, {'l1.hidden state': h1.data[0], 'l2.hidden state': h2.data[0], 'l2.cell state': c2.data[0], 'l2.cell input': a2.squeeze(),
                           'l2.input gates': i2.squeeze(), 'l2.forget gates': f2.squeeze(), 'l2.output gates': o2.squeeze()}
        else:
            return pi, v

    def reset(self):
        self.l2.reset_state()

    def get_persistent(self):
        return [self.l2.c, self.l2.h]

    def set_persistent(self, _c, _h):
        self.l2.c = _c
        self.l2.h = _h

    def unchain_backward(self):
        if not self.l2.h is None:
            self.l2.h.unchain_backward()
        if not self.l2.c is None:
            self.l2.c.unchain_backward()

### MODELS WITH CONTINUOUS ACTIONS AS OUTPUT ###

class GaussianMLP(Chain):
    """
    Model implements actor and critic with shared model
    MLP variant; supports multiple frames as input
    """

    def __init__(self, ninput, nhidden, noutput, nframes=1, covariance = 'fixed'):
        """

        :param ninput:
        :param nhidden:
        :param noutput: number of action outputs
        """

        self.covariance = covariance

        if covariance == 'fixed':

            super(GaussianMLP, self).__init__(
                l1=L.Linear(nframes * ninput, nhidden, initialW=init.HeNormal()),
                mu=L.Linear(nhidden, noutput, initialW=init.HeNormal()),
                v=L.Linear(nhidden, 1, initialW=init.HeNormal()),
            )

        elif covariance == 'spherical':

            super(GaussianMLP, self).__init__(
                l1=L.Linear(nframes * ninput, nhidden, initialW=init.HeNormal()),
                mu=L.Linear(nhidden, noutput, initialW=init.HeNormal()),
                lsigma2=L.Linear(nhidden, 1, initialW=init.Zero()),
                v=L.Linear(nhidden, 1, initialW=init.HeNormal()),
            )

        else:  # diagonal covariance

            super(GaussianMLP, self).__init__(
                l1=L.Linear(nframes * ninput, nhidden, initialW=init.HeNormal()),
                mu=L.Linear(nhidden, noutput, initialW=init.HeNormal()),
                lsigma2=L.Linear(nhidden, noutput, initialW=init.HeNormal()),
                v=L.Linear(nhidden, 1, initialW=init.HeNormal()),
            )

        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput
        self.nframes = nframes

        self.reset()

        # Note that pi and v define the actor and critic, respectively

    def __call__(self, x, persistent=False, internal_states=False):
        """

        :param x: sensory input
        :param persistent: whether or not to retain the internal state (if any)
        :param internal_states: whether or not to return internal states
        :return: policy output pi and value v
        """

        # add stimulus to buffer
        if self.nframes == 1:
            h = F.relu(self.l1(Variable(x)))
        else:

            if persistent:
                _buffer = self.buffer

            if self.idx >= self.nframes - 1:
                self.buffer = np.vstack([self.buffer[1:self.nframes], x])
            else:
                self.buffer[self.idx] = x

            h = F.relu(self.l1(Variable(self.buffer.reshape([1, self.buffer.size]))))

            if persistent:
                self.buffer = _buffer
            else:
                self.idx += 1

        if self.covariance == 'fixed':

            mu = self.mu(h)
            lsigma2 = Variable(np.zeros([1, self.noutput]).astype('float32'))
            v = self.v(h)

        else:

            mu = self.mu(h)
            lsigma2 = self.lsigma2(h)
            v = self.v(h)

        if internal_states:
            return [mu, lsigma2], v, {'hidden state': h.data[0]}
        else:
            return [mu, lsigma2], v

    def unchain_backward(self):
        pass


    def reset(self):

        if self.nframes > 1:
            self.buffer = np.zeros([self.nframes, self.ninput], dtype=np.float32)
            self.idx = 0


    def get_internal(self):
        """
        Returns: internal states
        """


class GaussianRNN(Chain):
    """
    Model implements actor and critic with shared model and continuous action space
    """

    def __init__(self, ninput, nhidden, noutput, covariance = 'fixed'):
        """

        :param ninput:
        :param nhidden:
        :param noutput: number of action outputs
        :param covariance: one of fixed, spherical, diagonal
        """

        self.covariance = covariance

        if covariance == 'fixed':

            super(GaussianRNN, self).__init__(
                l1=L.LSTM(ninput, nhidden),
                mu=L.Linear(nhidden, noutput, initialW=init.HeNormal()),
                v=L.Linear(nhidden, 1, initialW=init.HeNormal()),
            )

        elif covariance == 'spherical':

            super(GaussianRNN, self).__init__(
                l1=L.LSTM(ninput, nhidden),
                mu=L.Linear(nhidden, noutput, initialW=init.HeNormal()),
                lsigma2=L.Linear(nhidden, 1, initialW=init.HeNormal()),
                v=L.Linear(nhidden, 1, initialW=init.HeNormal()),
            )

        else: # diagonal covariance

            super(GaussianRNN, self).__init__(
                l1=L.LSTM(ninput, nhidden),
                mu=L.Linear(nhidden, noutput, initialW=init.HeNormal()),
                lsigma2=L.Linear(nhidden, noutput, initialW=init.HeNormal()),
                v=L.Linear(nhidden, 1, initialW=init.HeNormal()),
            )

        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput

        # Note that (mu,sigma2) and v define the actor and critic, respectively

    def __call__(self, x, persistent=False, internal_states=False):
        """

        :param x: sensory input
        :param persistent: whether or not to retain the internal state (if any)
        :param internal_states: whether or not to return internal states
                h = hidden states
                c = cell states
                a = cell input
                i = input gates
                f = forget gates
                o = output gates

        c &= \\tanh(a) \\text{sigmoid}(i)
           + c_{\\text{prev}} \\text{sigmoid}(f), \\\\
        h &= \\tanh(c) \\text{sigmoid}(o).

        http://www.felixgers.de/papers/phd.pdf
        http://docs.chainer.org/en/stable/_modules/chainer/functions/activation/lstm.html

        :return: policy output pi and value v

        """

        if persistent:
            _c, _h = self.get_persistent()

        if self.covariance == 'fixed':

            h = self.l1(Variable(x))
            mu = self.mu(h)
            lsigma2 = Variable(np.zeros([1, self.noutput]).astype('float32'))
            v = self.v(h)

        else:

            h = self.l1(Variable(x))
            mu = self.mu(h)
            lsigma2 = self.lsigma2(h)
            v = self.v(h)

        if internal_states:
            c = self.l1.c
            a, i, f, o = lstm._extract_gates(
                self.l1.upward.b.data.reshape(1, 4 * c.data.size, 1))

        if persistent:
            self.set_persistent(_c, _h)

        if internal_states:
            return [mu, lsigma2], v, {'hidden state': h.data[0], 'cell state': c.data[0], 'cell input': a.squeeze(),
                           'input gates': i.squeeze(), 'forget gates': f.squeeze(), 'output gates': o.squeeze()}
        else:
            return [mu, lsigma2], v

    def reset(self):
        self.l1.reset_state()

    def get_persistent(self):
        return [self.l1.c, self.l1.h]

    def set_persistent(self, _c, _h):
        self.l1.c = _c
        self.l1.h = _h

    def unchain_backward(self):
        if not self.l1.h is None:
            self.l1.h.unchain_backward()
        if not self.l1.c is None:
            self.l1.c.unchain_backward()