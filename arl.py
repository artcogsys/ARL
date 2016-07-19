import os
import numpy as np
import random
import copy
import multiprocessing as mp
from environments import *
from agents import *
from modelzoo import *
from asynchronous import *
from chainer import serializers

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

            # compute average loss
            # avg_loss = 0
            # for p in losses.keys():
            #     avg_loss = avg_loss + losses[p]
            # avg_loss = avg_loss/nprocs

            # return avg_loss

            # just return loss of the first worker
            return losses[losses.keys()[0]]

    def save(self, name):
        """
        Save model
        """

        serializers.save_npz('models/{0}.model'.format(name), self.agent.model)

    def load(self, name):
        """
        Load model
        """

        serializers.load_npz('models/{0}.model'.format(name), self.agent.model)

    def run(self, test_iter):
        """

        Args:
            test_iter: number of test iterations

        Returns:
            rewards: reward for each time point
            ground_truth: ground truth state for each time point
            observations: observation for each time point
            actions: selected action for each time point
            terminal: whether or not we are in a terminal (done) state for each time point
            log_prob: Output of the actor (log probability of selected action)
            entropy: Entropy of the stochastic policy according to the actor
            value: Output of the critic (estimated value according to current state of the model)
            internal: internal states (hidden units)
        """

        ground_truth = np.zeros([test_iter, 1], dtype=np.float32)

        observations = np.zeros(np.hstack([test_iter, self.agent.ninput]), dtype=np.float32)
        observations[:] = np.nan

        actions = np.zeros([test_iter, 1], dtype=np.uint8)
        actions[:] = np.nan

        rewards = np.zeros([test_iter, 1], dtype=np.float32)
        rewards[:] = np.nan

        log_prob = np.zeros([test_iter, 1], dtype=np.float32)
        log_prob[:] = np.nan

        entropy = np.zeros([test_iter, 1], dtype=np.float32)
        entropy[:] = np.nan

        value = np.zeros([test_iter, 1], dtype=np.float32)
        value[:] = np.nan

        done = np.zeros([test_iter, 1], dtype=np.bool)

        internal = []

        ###
        # Start run

        env = self.agent.environment

        # initialize environment; this generates a ground truth
        obs = env.reset()
        observations[0] = obs

        ground_truth[0] = env.get_ground_truth()

        # reset agent
        self.agent.model.reset()

        for i in xrange(test_iter):

            # generate action using actor model
            actions[i], pi, v = self.agent.act(obs)

            p = F.softmax(pi)
            action = np.random.choice(env.noutput, None, True, p.data[0])
            actions[i] = action
            logp = F.log_softmax(pi)
            log_prob[i] = F.select_item(logp, Variable(np.asarray([action], dtype=np.int32))).data
            entropy[i] = - F.sum(p * logp, axis=1).data

            # value according to critic
            value[i] = v.data

            # Perform action and receive new observations and reward
            obs, rewards[i], done[i] = env.step(actions[i])

            if i < test_iter - 1:
                observations[i + 1] = obs
                ground_truth[i + 1] = env.get_ground_truth()

            # if done:
            #     self.model.reset()


        return rewards, ground_truth, observations, actions, done, log_prob, entropy, value, internal


    def simulate(self, ground_truth, observations, actions):
        """

        Args:
                ground_truth: of form np.array([test_iter, n_ground_truth], dtype = np.float32)
                observations: of form np.array([test_iter, n_ground_truth], dtype = np.float32)
                actions: of form np.array([test_iter, n_ground_truth], dtype = np.float32)
                internal_states: whether or not to return internal states

        Returns:
            rewards: reward for each time point
            terminal: whether or not we are in a terminal (done) state for each time point
            log_prob: Output of the actor (log probability of selected action)
            entropy: Entropy of the stochastic policy according to the actor
            value: Output of the critic (estimated value according to current state of the model)
            internal: internal states (hidden units)
        """

        test_iter = ground_truth.shape[0]

        rewards = np.zeros([test_iter, 1], dtype=np.float32)
        rewards[:] = np.nan

        log_prob = np.zeros([test_iter, 1], dtype=np.float32)
        log_prob[:] = np.nan

        entropy = np.zeros([test_iter, 1], dtype=np.float32)
        entropy[:] = np.nan

        value = np.zeros([test_iter, 1], dtype=np.float32)
        value[:] = np.nan

        done = np.zeros([test_iter, 1], dtype=np.bool)

        internal = []

        ###
        # Start run

        env = self.agent.environment

        # initialize environment
        obs = env.reset()
        obs_shape = obs.shape
        obs = observations[0].reshape(obs_shape)

        # force ground truth to be the same as that of the behavioural data
        env.set_ground_truth(ground_truth[0])

        # reset agent
        self.agent.model.reset()

        for i in xrange(test_iter):

            # generate action using actor model
            action, pi, v = self.agent.act(obs)

            p = F.softmax(pi)
            logp = F.log_softmax(pi)
            log_prob[i] = F.select_item(logp, Variable(np.asarray([actions[i,0]], dtype=np.int32))).data
            entropy[i] = - F.sum(p * logp, axis=1).data

            # value according to critic
            value[i] = v.data

            # Perform action and receive new observations and reward
            _, rewards[i], done[i] = env.step(actions[i])

            # For the last step we don't have an observation or ground truth (end of experiment)
            if i < test_iter - 1:
                env.set_ground_truth(ground_truth[i+1])
                obs = observations[i+1].reshape(obs_shape)

            # if done:
            #     self.model.reset()

        return rewards, done, log_prob, entropy, value, internal