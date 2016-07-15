from chainer import Chain, Variable
import chainer.initializers as init
import chainer.functions as F
import chainer.links as L
import numpy as np

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
            v=L.Linear(nhidden, 1, initialW=init.HeNormal()),
        )

        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput
        self.nframes = nframes

        self.reset()

        # Note that pi and v define the actor and critic, respectively

    def __call__(self, x, persistent=False):
        """

        :param x: sensory input
        :param persistent: whether or not to retain the internal state (if any)
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

        return pi, v


    def unchain_backward(self):
        pass


    def reset(self):

        if self.nframes > 1:
            self.buffer = np.zeros([self.nframes, self.ninput], dtype=np.float32)
            self.idx = 0

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

    def __call__(self, x, persistent = False):
        """

        :param x: sensory input (1 x nframes x ninput[0] x ninput[1])
        :param persistent: whether or not to retain the internal state (if any)
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

        # Note that pi and v define the actor and critic, respectively

    def __call__(self, x, persistent=False):
        """

        :param x: sensory input
        :param persistent: whether or not to retain the internal state (if any)
        :return: policy output pi and value v

        """

        if persistent:
            _c, _h = self.get_persistent()
            h = self.l1(Variable(x))
            pi = self.pi(h)
            v = self.v(h)
            self.set_persistent(_c, _h)
        else:
            h = self.l1(Variable(x))
            pi = self.pi(h)
            v = self.v(h)

        return pi, v

    def reset(self):
        self.l1.reset_state()

    def get_persistent(self):
        return [self.l1.c, self.l1.h]

    def set_persistent(self, _c, _h):
        self.l1.c = _c
        self.l1.h = _h

    def unchain_backward(self):
        self.l1.h.unchain_backward()
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

    def __call__(self, x, persistent=False):
        """

        :param x: sensory input
        :param persistent: whether or not to retain the internal state (if any)
        :return: policy output pi and value v

        """

        if persistent:
            _c, _h = self.get_persistent()
            h = F.relu(self.l1(Variable(np.expand_dims(x, axis=0))))
            h = self.l2(h)
            pi = self.pi(h)
            v = self.v(h)
            self.set_persistent(_c, _h)
        else:
            h = F.relu(self.l1(Variable(np.expand_dims(x, axis=0))))
            h = self.l2(h)
            pi = self.pi(h)
            v = self.v(h)

        return pi, v

    def reset(self):
        self.l2.reset_state()

    def get_persistent(self):
        return [self.l2.c, self.l2.h]

    def set_persistent(self, _c, _h):
        self.l2.c = _c
        self.l2.h = _h

    def unchain_backward(self):
        self.l2.h.unchain_backward()
        self.l2.c.unchain_backward()