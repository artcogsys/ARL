import numpy as np
import chainer
import multiprocessing as mp

###########
# Helper functions to read and write shared parameters

def set_shared_params(a, b):
    """
    Args:
      a (chainer.Link): link whose params are to be replaced
      b (dict): dict that consists of (param_name, multiprocessing.Array)
    """
    assert isinstance(a, chainer.Link)
    for param_name, param in a.namedparams():
        if param_name in b:
            shared_param = b[param_name]
            param.data = np.frombuffer(
                shared_param, dtype=param.data.dtype).reshape(param.data.shape)

def set_shared_states(a, b):
    assert isinstance(a, chainer.Optimizer)
    assert hasattr(a, 'target'), 'Optimizer.setup must be called first'
    for state_name, shared_state in b.items():
        for param_name, param in shared_state.items():
            old_param = a._states[state_name][param_name]
            a._states[state_name][param_name] = np.frombuffer(
                param,
                dtype=old_param.dtype).reshape(old_param.shape)

def extract_params_as_shared_arrays(link):
    assert isinstance(link, chainer.Link)
    shared_arrays = {}
    for param_name, param in link.namedparams():
        shared_arrays[param_name] = mp.RawArray('f', param.data.ravel())
    return shared_arrays

def share_params_as_shared_arrays(link):
    shared_arrays = extract_params_as_shared_arrays(link)
    set_shared_params(link, shared_arrays)
    return shared_arrays

def share_states_as_shared_arrays(link):
    shared_arrays = extract_states_as_shared_arrays(link)
    set_shared_states(link, shared_arrays)
    return shared_arrays

def extract_states_as_shared_arrays(optimizer):
    assert isinstance(optimizer, chainer.Optimizer)
    assert hasattr(optimizer, 'target'), 'Optimizer.setup must be called first'
    shared_arrays = {}
    for state_name, state in optimizer._states.items():
        shared_arrays[state_name] = {}
        for param_name, param in state.items():
            shared_arrays[state_name][
                param_name] = mp.RawArray('f', param.ravel())
    return shared_arrays


class RMSpropAsync(chainer.optimizer.GradientMethod):
    """

    RMSprop for asynchronous methods.

    The only difference from chainer.optimizers.RMSprop in that the epsilon is
    outside the square root.

    Small non-significant update as listed in supplement of mnih paper

    """

    def __init__(self, lr=0.01, alpha=0.99, eps=0.1):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps

    def init_state(self, param, state):
        xp = chainer.cuda.get_array_module(param.data)
        state['ms'] = xp.zeros_like(param.data)

    def update_one_cpu(self, param, state):
        ms = state['ms']
        grad = param.grad

        ms *= self.alpha
        ms += (1 - self.alpha) * grad * grad
        param.data -= self.lr * grad / np.sqrt(ms + self.eps)

    def update_one_gpu(self, param, state):
        chainer.cuda.elementwise(
            'T grad, T lr, T alpha, T eps',
            'T param, T ms',
            '''ms = alpha * ms + (1 - alpha) * grad * grad;
               param -= lr * grad / sqrt(ms + eps);''',
            'rmsprop')(param.grad, self.lr, self.alpha, self.eps,
                       param.data, state['ms'])
