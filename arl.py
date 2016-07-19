import os
import numpy as np
import random
import copy
import multiprocessing as mp
from environments import *
from agents import *
from modelzoo import *
from asynchronous import *

###########
# Asynchronous reinforcement learning

class ARL(object):
    """
    Asynchronous reinforcement learning
    """

    def __init__(self, agent, **kwargs):
        """

        :param: agent: a learning agent
        :param: learning_rates: initial learning rates to choose from for each process
        """

        self.agent = agent

        self.optimizer = kwargs.get('optimizer', RMSpropAsync())
        self.optimizer.setup(self.agent.model)

        self.learning_rates = kwargs.get('learning_rates', np.logspace(-4,-2,100))

    def learn(self, niter, nprocs = None):
        """
        Learn model parameters based on asynchronous updates of multiple agents
        :param niter: number of training iterations
        :param nprocesses: number of parallel processes; number of CPU nodes when left unspecified
                           if nprocs = 1 then we run just a single threaded implementation

        :return: loss
        """

        if nprocs == 1:

            loss = self.agent.learn(niter)

            return loss

        else:

            # Prevent numpy from using multiple threads
            os.environ['OMP_NUM_THREADS'] = '1'

            # here shared model parameters are defined
            shared_params = share_params_as_shared_arrays(self.agent.model)
            shared_states = share_states_as_shared_arrays(self.optimizer)

            def run_func(in_q, out_q, niter):
                """
                Multiprocessing workers operate on stateless functions. We pass objects by adding them to a queue
                :param: in_q : input queue containing to-be-passed objects
                :param: out_q : output queue containing the worker-specific loss
                """

                # set random seed
                seed = np.random.randint(0, 2 ** 32)
                random.seed(seed)
                np.random.seed(seed)

                # retrieve pickled objects from input queue
                model = in_q.get()
                agent = in_q.get()
                optimizer = in_q.get()

                # set shared model parameters
                set_shared_params(model, shared_params)

                # set shared optimizer parameters
                set_shared_states(optimizer, shared_states)

                # assign shared model to agent
                agent.shared_model = model

                # define process-specific model
                agent.model = copy.deepcopy(agent.shared_model)

                agent.optimizer = optimizer
                agent.optimizer.setup(agent.shared_model)

                # Agent name is equated with index
                agent.name = mp.current_process().name

                loss = agent.learn(niter)

                out_q.put({mp.current_process().name : loss})

            # define number of parallel processes
            if not nprocs:
                nprocs = mp.cpu_count()
                print 'Using {0} CPU cores'.format(nprocs)

            # define output queue
            out_q = mp.Queue()

            # determine learning rates for each process
            sz = self.learning_rates.size
            n = sz / nprocs
            lrates = self.learning_rates[0:sz:n]

            # run parallel learning scheme
            procs = []
            for i in range(nprocs):

                # define separate input queue for each worker
                in_q = mp.Queue()
                in_q.put(self.agent.model)
                in_q.put(self.agent)

                # each optimizer gets its own initial learning rate
                optimizer = self.optimizer
                optimizer.lr = lrates[i]
                in_q.put(self.optimizer)

                p = mp.Process(target=run_func, args=(in_q, out_q, niter))
                procs.append(p)
                p.start()

            # Collect all results into a single result dict
            losses = {}
            for i in range(nprocs):
                losses.update(out_q.get())

            for p in procs:
                p.join()

            # return losses of all workers
            return losses, self.agent
