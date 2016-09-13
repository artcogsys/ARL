import math
import numpy as np
from chainer import optimizers, Variable
import chainer.functions as F
from asynchronous import RMSpropAsync
from chainer import serializers
from chainer import variable
from chainer.functions.math import exponential
from chainer.functions.math import sum
import matplotlib.pyplot as plt

# set interactive mode
plt.ion()

def copy_param(target_link, source_link):
    """Copy parameters of a link to another link.
    """
    target_params = dict(target_link.namedparams())
    for param_name, param in source_link.namedparams():
        target_params[param_name].data[:] = param.data


def copy_grad(target_link, source_link):
    """Copy gradients of a link to another link.
    """
    target_params = dict(target_link.namedparams())
    for param_name, param in source_link.namedparams():
        target_params[param_name].grad[:] = param.grad

###
# Base class for an agent

class Agent(object):
    """
    Base class for any agent
    """

    def __init__(self, environment, **kwargs):
        """
        An agent is always defined relative to an environment

        :param environment:
        :param kwargs:
        """

        # define environment
        self.environment = environment

        self.ninput = environment.ninput
        self.noutput = environment.noutput

        # define optimizer
        self.optimizer = kwargs.get('optimizer', RMSpropAsync())

        # define file name
        self.file_name = kwargs.get('file_name', 'temp')

        # agent name
        self.name = kwargs.get('name', 'Agent')

        # discounting factor
        self.gamma = kwargs.get('gamma', 0.99)

    def act(self, observation):
        return np.random.randint(self.noutput)

    def learn(self, niter):
        return [np.nan]

    def save(self, name):
        """
        Save model
        """

        serializers.save_npz('models/{0}.model'.format(name), self.model)

    def load(self, name):
        """
        Load model
        """

        serializers.load_npz('models/{0}.model'.format(name), self.model)


class A2C(Agent):
    """

    Advantage Actor Critic model

    See:
    http://www.rage.net/~greg/2016-07-05-ActorCritic-with-OpenAI-Gym.html
    Mnih paper

    """

    def __init__(self, environment, model, **kwargs):
        super(A2C, self).__init__(environment, **kwargs)

        # the actor-critic model
        self.model = model
        self.optimizer.setup(self.model)

        # the global shared model
        self.shared_model = None

        # define discrete or continuous action space
        self.discrete = kwargs.get('discrete', True)

        # maximal number of accumulation steps
        self.t_max = 10

        # entropy parameter
        if self.discrete:
            self.beta = 1e-2
        else:
            self.beta = 1e-4

        # stored variables needed for gradient updating
        self.past_score_function = {}
        self.past_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}

    def learn(self, niter, callback = None):
        """

        This implementation follow advantage actor-critic Algorithm 3 in Mnih as closely as possible

        Args:
            niter: number of training iterations (T_max in A3C paper)
            callback: callback function to test convergence properties (None, self.callback or custom function)

        Returns: loss and times at which loss was acquired

        """

        # reset the environment and get observation
        obs = self.environment.reset()

        # create trajectory of learning rates
        learning_rates = np.linspace(self.optimizer.lr, 0, niter, False)

        # outer training loop
        pi_losses = []
        v_losses = []
        ts = []
        t_start = t = 0
        while True:

            # update from shared model
            if self.shared_model:
                copy_param(target_link=self.model, source_link=self.shared_model)

            if t > 0:
                self.model.unchain_backward()

            # running index
            t_start = t

            while True:

                idx = t - t_start

                # store observation
                self.past_states[idx] = obs

                # generate action using actor model
                action, pi, v = self.act(obs)

                # store log policy data
                self.past_score_function[idx] = self.score_function(action, pi)

                # compute entropy
                self.past_entropy[idx] = self.entropy(pi)

                # perform action via actor and receive new observations and reward
                obs, reward, done = self.environment.step(action)

                # if done:
                #     self.model.reset()

                # No reward clipping; will lead to random guessing as best option for our environment
                # reward = np.clip(reward, -1, 1)

                # store reward and value
                self.past_rewards[idx] = reward
                self.past_values[idx] = v

                t += 1

                if done or idx == (self.t_max - 1):
                    break

            if done:
                R = 0
            else:
                _, vout = self.model(obs, persistent = True)
                R = float(vout.data)

            pi_loss = v_loss = 0
            for i in range(idx, -1, -1):

                R = self.past_rewards[i] + self.gamma * R

                v = self.past_values[i]

                # Compute advantage (difference between approximation of action-value and value)
                advantage = R - v

                # get log probability
                score_function = self.past_score_function[i]

                # Compute entropy
                entropy = self.past_entropy[i]

                # Log policy (probability of action given observations) is increased proportionally to advantage
                pi_loss -= score_function * float(advantage.data)

                # loss is reduced by high entropy (stochastic) policies
                pi_loss -= self.beta * entropy

                # Get squared difference between accumulated reward and value function
                v_loss += advantage ** 2

            # Perform (a)synchronous updating

            v_loss = F.reshape(v_loss, pi_loss.data.shape)

            # Compute total loss
            # 0.5 supposedly used by Mnih et al
            loss = pi_loss + 0.5 * v_loss

            # Normalization of the loss of sequences truncated by terminal states
            # NOTE: This causes instabilities; may be solvable by scaling learning rate appropriately
            # factor = self.t_max / (idx + 1)
            # loss *= factor

            # Compute gradients using thread-specific model
            self.model.zerograds()
            loss.backward()

            # Copy the gradients to the globally shared model
            if self.shared_model:
                self.shared_model.zerograds()
                copy_grad(target_link=self.shared_model, source_link=self.model)

            # update the shared model
            self.optimizer.update()

            # reset stored values
            self.past_score_function = {}
            self.past_entropy = {}
            self.past_states = {}
            self.past_rewards = {}
            self.past_values = {}

            # after next line???
            if t >= niter:
                break

            pi_losses.append(pi_loss.data[0])
            v_losses.append(v_loss.data[0])

            ts.append(t)

            # callback during training
            if callback is not None:
                callback(self.name, t, pi_losses, v_losses, action, pi, v, reward)
            else:
                self.callback_learn(self.name, t, pi_losses, v_losses, action, pi, v, reward)

        return [ts, pi_losses, v_losses]

    def act(self, obs, internal_states = False):
        """

        Advantage actor-critic is always on-policy

        :param obs: sensory input
        :return: action
        """

        # get output of actor model for this observation
        if internal_states:
            pi, v, internal = self.model(obs, internal_states=True)
        else:
            pi, v = self.model(obs)

        # generate action for discrete or continuous action space
        if self.discrete:

            # generate action according to policy
            p = F.softmax(pi).data[0]

            # normalize p in case tiny floating precision problems occur
            assert( np.sum(p) > 0.999999 )
            p = p.astype('float64')
            p /= p.sum()

            action = np.random.choice(self.noutput, None, True, p)

        else:

            mu = pi[0].data[0]

            # softplus variant
            sigma2 = F.softplus(pi[1])
            C = sigma2.data[0] * np.eye(mu.size)
            action = np.random.multivariate_normal(mu, C)

            # exponential variant
            # sigma2 = F.exp(pi[1])
            # C = sigma2.data[0] * np.eye(mu.size)
            # action = np.random.multivariate_normal(mu, C)

            # action = mu

        # return action chosen according to stochastic policy
        if internal_states:
            return action, pi, v, internal
        else:
            return action, pi, v

    def entropy(self,pi):
        """

        Args:
            pi: stochastic policy

        Returns: entropy of the stochastic policy

        """

        if self.discrete:

            p = F.softmax(pi)
            logp = F.log_softmax(pi)

            return - F.sum(p * logp, axis=1)

        else:

            # softplus variant
            sigma2 = F.softplus(pi[1])
            return - F.sum(0.5 * F.log(2 * math.pi * sigma2) + 1, axis=1)

            # exponential variant
            # return - F.sum(0.5 * np.log(2 * math.pi) + 0.5 * pi[1] + 1, axis=1)

    def score_function(self, action, pi):
        """

        Args:
            action: selected action
            pi: stochastic policy

        Returns: the log of the policy (the gradient is computed automatically :o) )

        """

        if self.discrete:

            logp = F.log_softmax(pi)
            return F.select_item(logp, Variable(np.asarray([action], dtype=np.int32)))

        else:


            # exponential variant
            # mu = pi[0]
            # log_sigma2 = pi[1]
            # sigma2 = F.exp(pi[1])
            # a = Variable(np.asarray([action], dtype=np.float32))
            # a_dif = a - mu
            # v = -0.5 * F.sum(log_sigma2) -0.5 * F.sum(1.0/sigma2 * a_dif * a_dif)
            # return F.expand_dims(v,0)

            # softplus variant
            mu = pi[0]
            sigma2 = F.softplus(pi[1])
            log_sigma2 = F.log(sigma2)
            a = Variable(np.asarray([action], dtype=np.float32))
            a_dif = a - mu
            v = -0.5 * F.sum(log_sigma2) -0.5 * F.sum(1.0/sigma2 * a_dif * a_dif)
            return F.expand_dims(v,0)

            # using F.gaussian_nll
            # # sigma2 = F.softplus(pi[1])  # needed for numerical stability
            # # log_var = F.log(sigma2)  # expected by gaussian_nll
            # # v = - F.gaussian_nll(Variable(np.asarray([action], dtype=np.float32)), mu, log_var)



    def simulate(self, test_iter):
        """

        Args:
            test_iter: number of test iterations

        Returns:
            rewards: reward for each time point
            ground_truth: ground truth state for each time point
            observations: observation for each time point
            actions: selected action for each time point
            done: whether or not we are in a terminal (done) state for each time point
        """

        # set environment

        env = self.environment

        # assign variables

        ground_truth = np.zeros([test_iter, env.nstates], dtype=np.float32)

        observations = np.zeros(np.hstack([test_iter, self.ninput]), dtype=np.float32)
        observations[:] = np.nan

        if self.discrete:
            actions = np.zeros([test_iter, env.naction], dtype=np.uint8)
        else:
            actions = np.zeros([test_iter, env.naction], dtype=np.float32)
        actions[:] = np.nan

        rewards = np.zeros([test_iter, 1], dtype=np.float32)
        rewards[:] = np.nan

        done = np.zeros([test_iter, 1], dtype=np.bool)

        ###
        # Start run

        # initialize environment; this generates a ground truth
        obs = env.reset()
        observations[0] = obs

        ground_truth[0] = env.get_ground_truth()

        # reset agent
        self.model.reset()

        for i in xrange(test_iter):

            # generate action using actor model
            actions[i], pi, v = self.act(obs)

            # Perform action and receive new observations and reward
            obs, rewards[i], done[i] = env.step(actions[i])

            if i < test_iter - 1:
                observations[i + 1] = obs
                ground_truth[i + 1] = env.get_ground_truth()

            # if done[i]:
            #     self.model.reset()

        return rewards, ground_truth, observations, actions, done


    def analyze(self, sim_rewards, ground_truth, observations, actions, callback):
        """

        Args:
            sim_rewards: rewards that are computed during simulation; only used for sanity checking
            ground_truth: of form np.array([test_iter, n_ground_truth], dtype = np.float32)
            observations: of form np.array([test_iter, n_ground_truth], dtype = np.float32)
            actions: of form np.array([test_iter, n_ground_truth], dtype = np.float32)

        Returns:
            rewards: Can be used for sanity checking (does analysis yield same reward trajectory as experiment?)
            score_function: Output of the actor (log probability of selected action)
            entropy: Entropy of the stochastic policy according to the actor
            value: Output of the critic (estimated value according to current state of the model)
            returns: The expected return at each point in time
            advantage: returns - value
	        surprise: (V_t - r_t - V_t+1)^2
            _internal_states: internal states (hidden units, cell states, etc.; model dependent)
	        pstate: probabilistic state estimate according to generative model (not part of the neural net!)

        Note:
            Returns, advantage and surprise are all signals that depend on an n-step look-ahead. This is a signal which is
            not available to the network/subject at test time.

            We can assume that the network has access to the reward r_t. We also know that V_t = r_t + V_t+1
            So we can say that the surprise signal at time t+1 is given by (V_t - r_t - V_t+1)^2.


        """

        test_iter = ground_truth.shape[0]

        rewards = np.zeros([test_iter, 1], dtype=np.float32)
        rewards[:] = np.nan

        score_function = np.zeros([test_iter, 1], dtype=np.float32)
        score_function[:] = np.nan

        entropy = np.zeros([test_iter, 1], dtype=np.float32)
        entropy[:] = np.nan

        value = np.zeros([test_iter, 1], dtype=np.float32)
        value[:] = np.nan

        surprise = np.zeros([test_iter, 1], dtype=np.float32)
        surprise[:] = np.nan

        returns = np.zeros([test_iter, 1], dtype=np.float32)

        done = np.zeros([test_iter, 1], dtype=np.bool)

        pstate = np.zeros([test_iter, 1], dtype=np.float32)
        pstate[:] = np.nan

        _internal_states = {}

        ###
        # Start run

        env = self.environment

        # initialize environment
        obs = env.reset()
        obs_shape = obs.shape
        obs = observations[0].reshape(obs_shape)

        # force ground truth to be the same as that of the behavioural data
        env.set_ground_truth(ground_truth[0])

        # reset agent
        self.model.reset()

        for i in xrange(test_iter):

            pstate[i] = self.environment.get_pstate()

            # generate action using actor model
            action, pi, v, internal = self.act(obs, internal_states = True)

            # actor-related regressors
            score_function[i] = self.score_function(action, pi).data
            entropy[i] = self.entropy(pi).data

            # critic-related regressor
            value[i] = v.data

            # initialize internal states
            if i == 0:
                for k in internal.keys():
                    ksize = [test_iter] + list(internal[k].shape)
                    _internal_states[k] = np.zeros(ksize, dtype=np.float32)

            # add internal states
            for k in internal.keys():
                _internal_states[k][i] = internal[k]

            # perform action via actor and receive new observations and reward
            obs, rewards[i], done[i] = self.environment.step(actions[i])

            # compete surprise signal (V_t - r_t - V_t+1)^2
            if i > 0:
                surprise[i] = (value[i-1] - rewards[i-1] - value[i])**2

            # if done:
            #     self.model.reset()

            # No reward clipping; will lead to random guessing as best option for our environment
            # reward = np.clip(reward, -1, 1)

            # For the last step we don't have an observation or ground truth (end of experiment)
            if i < test_iter - 1:
                env.set_ground_truth(ground_truth[i + 1])
                obs = observations[i + 1].reshape(obs_shape)

        # if done:
        #     R = 0
        # else:
        #     _, vout = self.model(obs, persistent=True)
        #     R = float(vout.data)
        #
        # for i in range(test_iter-1, -1, -1):
        #
        #     R = rewards[i] + self.gamma * R
        #     returns[i] = R

        # compute value associated with the last produced observation
        _, vout = self.model(obs, persistent=True)
        vlast = float(vout.data)

        # stuff below are all regressors that require an n-step lookahead!

        # compute n-step reward as approximation of action-value function for each time point
        for i in range(test_iter):

            # take at most n steps
            for j in range(self.t_max):

                idx = i + j

                if idx == test_iter:
                    returns[i] += self.gamma ** (idx-i) * vlast
                    break

                if idx > i and done[idx]:
                    break

                if j == self.t_max - 1:
                    returns[i] += self.gamma ** (idx-i) * value[idx]
                else:
                    returns[i] += self.gamma ** (idx-i) * rewards[i]

        advantage = returns - value

        # sanity check so we are sure that analyze is ran on the exact same sequence as simulate
        assert ((rewards == sim_rewards).all())

        # callback of analysis function
        if callback is not None:
            callback(self.file_name, ground_truth, actions, rewards, score_function, entropy, value, returns, advantage, surprise, _internal_states, pstate)
        else:
            self.callback_analyze(self.file_name, ground_truth, actions, rewards, score_function, entropy, value, returns, advantage, surprise, _internal_states, pstate)

    def render(self, ground_truth, observations, actions):
        """
        Renders the perception action loop for visual inspection (UNFINISHED)

        Args:
            ground_truth:
            observations:
            actions:

        Returns:

        """
        test_iter = ground_truth.shape[0]

        ###
        # Start run

        env = self.environment

        # initialize environment
        obs = env.reset()
        obs_shape = obs.shape
        obs = observations[0].reshape(obs_shape)

        # force ground truth to be the same as that of the behavioural data
        env.set_ground_truth(ground_truth[0])

        # reset agent
        self.model.reset()

        for i in xrange(test_iter):

            # generate action using actor model
            action, pi, v, internal = self.act(obs)

            # render state, observation and action
            self.environment.render(action)

            # perform action via actor and receive new observations and reward
            obs, reward, done = self.environment.step(actions[i])

            # if done:
            #     self.model.reset()

            # No reward clipping; will lead to random guessing as best option for our environment
            # reward = np.clip(reward, -1, 1)

            # For the last step we don't have an observation or ground truth (end of experiment)
            if i < test_iter - 1:
                env.set_ground_truth(ground_truth[i + 1])
                obs = observations[i + 1].reshape(obs_shape)


    def callback_learn(self, name, t, pi_losses, v_losses, action, pi, v, reward):
        """
        Default callback function to check properties during learning

        Args:
            name: name of the agent
            t: time step
            losses: loss trajectory
            pi: current policy output
            v: current value output
            reward: current reward

        Returns:

        """

        # rough indication of how the loss changes
        print '{0}; {1}; {2:03.5f}'.format(t, name, pi_losses[-1]+0.5*v_losses[-1])

    def callback_analyze(self, file_name, ground_truth, actions, rewards, score_function, entropy, value, returns, advantage, surprise, _internal_states, pstate):

        ##########
        # visualize results

        # plot rewards

        t = range(len(rewards))

        plt.clf()
        plt.plot(t, np.cumsum(rewards), 'k')
        plt.xlabel('iteration')
        plt.ylabel('cumulative reward')
        plt.savefig('figures/' + file_name + '__reward.png')

        plt.clf()
        plt.plot(t, score_function, 'k')
        plt.xlabel('iteration')
        plt.ylabel('score_function')
        plt.savefig('figures/' + file_name + '__score_function.png')

        plt.clf()
        plt.plot(t, entropy, 'k')
        plt.xlabel('iteration')
        plt.ylabel('entropy')
        plt.savefig('figures/' + file_name + '__entropy.png')

        plt.clf()
        plt.plot(t, value, 'k')
        plt.xlabel('iteration')
        plt.ylabel('value')
        plt.savefig('figures/' + file_name + '__value.png')

        plt.clf()
        plt.plot(t, returns, 'k')
        plt.xlabel('iteration')
        plt.ylabel('returns')
        plt.savefig('figures/' + file_name + '__returns.png')

        plt.clf()
        plt.plot(t, advantage, 'k')
        plt.xlabel('iteration')
        plt.ylabel('advantage')
        plt.savefig('figures/' + file_name + '__advantage.png')

        plt.clf()
        plt.plot(t, surprise, 'k')
        plt.xlabel('iteration')
        plt.ylabel('surprise')
        plt.savefig('figures/' + file_name + '__surprise.png')
        plt.close()

        plt.clf()
        plt.plot(t, pstate, 'k')
        plt.xlabel('iteration')
        plt.ylabel('pstate')
        plt.savefig('figures/' + file_name + '__pstate.png')
        plt.close()