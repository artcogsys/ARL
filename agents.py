import numpy as np
from chainer import optimizers, Variable
import chainer.functions as F
from asynchronous import RMSpropAsync
from chainer import serializers

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

        # maximal number of accumulation steps
        self.t_max = 10

        # entropy parameter
        self.beta = 1e-2

        # stored variables needed for gradient updating
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}

    def learn(self, niter):
        """

        This implementation follow advantage actor-critic Algorithm 3 in Mnih as closely as possible

        Args:
            niter: number of training iterations (T_max in A3C paper)

        Returns: loss and times at which loss was acquired

        """

        # reset the environment and get observation
        obs = self.environment.reset()

        # create trajectory of learning rates
        learning_rates = np.linspace(self.optimizer.lr, 0, niter, False)

        # outer training loop
        losses = []
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

                self.past_action_log_prob[idx] = self.log_prob(action, pi)

                # perform action via actor and receive new observations and reward
                obs, reward, done = self.environment.step(action)

                # if done:
                #     self.model.reset()

                # No reward clipping; will lead to random guessing as best option for our environment
                # reward = np.clip(reward, -1, 1)

                # store reward and value
                self.past_rewards[idx] = reward
                self.past_values[idx] = v

                # compute entropy
                self.past_action_entropy[idx] = self.entropy(pi)

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

                # Compute advantage
                advantage = R - v

                # get log probability
                log_prob = self.past_action_log_prob[i]

                # Compute entropy
                entropy = self.past_action_entropy[i]

                # Log probability is increased proportionally to advantage
                pi_loss -= log_prob * float(advantage.data)

                # loss is reduced by high entropy (stochastic) policies
                pi_loss -= self.beta * entropy

                # Get squared difference between accumulated reward and value function
                v_loss += advantage ** 2

            # Perform (a)synchronous updating

            # Compute total loss
            # 0.5 supposedly used by Mnih et al
            loss = pi_loss + 0.5 * F.reshape(v_loss, pi_loss.data.shape)

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
            self.past_action_log_prob = {}
            self.past_action_entropy = {}
            self.past_states = {}
            self.past_rewards = {}
            self.past_values = {}

            if t >= niter:
                break

            # rough indication of how the loss changes
            print '{0}; {1}; {2:03.5f}'.format(t, self.name, loss.data[0])
            losses.append(loss.data[0])
            ts.append(t)

        return [ts, losses]

    def act(self, obs, internal_states = False):
        """

        Advantage actor-critic is always on-policy

        :param obs: sensory input
        :return: action
        """

        # get output of actor model for this observation
        if internal_states:
            pi, v, internal = self.model(obs, internal_states = True)
        else:
            pi, v = self.model(obs)

        # generate action according to policy
        p = F.softmax(pi).data[0]

        # normalize p in case tiny floating precision problems occur
        assert( np.sum(p) > 0.999999 )
        p = p.astype('float64')
        p /= p.sum()

        # return action chosen according to stochastic policy
        if internal_states:
            return np.random.choice(self.noutput, None, True, p), pi, v, internal
        else:
            return np.random.choice(self.noutput, None, True, p), pi, v

    def entropy(self,pi):
        """

        Args:
            pi: stochastic policy

        Returns: entropy of the stochastic policy

        """
        p = F.softmax(pi)
        logp = F.log_softmax(pi)

        return - F.sum(p * logp, axis=1)

    def log_prob(self, action, pi):
        """

        Args:
            action: selected action
            pi: stochastic policy

        Returns: log probability of selected action

        """

        logp = F.log_softmax(pi)
        return F.select_item(logp, Variable(np.asarray([action], dtype=np.int32)))

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

        ground_truth = np.zeros([test_iter, 1], dtype=np.float32)

        observations = np.zeros(np.hstack([test_iter, self.ninput]), dtype=np.float32)
        observations[:] = np.nan

        actions = np.zeros([test_iter, 1], dtype=np.uint8)
        actions[:] = np.nan

        rewards = np.zeros([test_iter, 1], dtype=np.float32)
        rewards[:] = np.nan

        done = np.zeros([test_iter, 1], dtype=np.bool)

        ###
        # Start run

        env = self.environment

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


    def analyze(self, ground_truth, observations, actions):
        """

        Args:
                ground_truth: of form np.array([test_iter, n_ground_truth], dtype = np.float32)
                observations: of form np.array([test_iter, n_ground_truth], dtype = np.float32)
                actions: of form np.array([test_iter, n_ground_truth], dtype = np.float32)

        Returns:
            rewards: Can be used for sanity checking (does analysis yield same reward trajectory as experiment?)
            log_prob: Output of the actor (log probability of selected action)
            entropy: Entropy of the stochastic policy according to the actor
            value: Output of the critic (estimated value according to current state of the model)
            returns: The expected return at each point in time
            advantage: returns - value
	        advantage_surprise: squared advantage as a possible measure for surprise
            _internal_states: internal states (hidden units, cell states, etc.; model dependent)

        Note: Squared advantage could be a measure of surprise?
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

        returns = np.zeros([test_iter, 1], dtype=np.float32)
        returns[:] = np.nan

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

            # generate action using actor model
            action, pi, v, internal = self.act(obs, internal_states = True)

            log_prob[i] = self.log_prob(action, pi).data

            # perform action via actor and receive new observations and reward
            obs, reward, done = self.environment.step(actions[i])

            # if done:
            #     self.model.reset()

            # No reward clipping; will lead to random guessing as best option for our environment
            # reward = np.clip(reward, -1, 1)

            # store reward and value
            rewards[i] = reward
            value[i] = v.data

            # compute entropy
            entropy[i] = self.entropy(pi).data

            # initialize internal states
            if i == 0:
                for k in internal.keys():
                    ksize = [test_iter] + list(internal[k].shape)
                    _internal_states[k] = np.zeros(ksize, dtype=np.float32)

            # add internal states
            for k in internal.keys():
                _internal_states[k][i] = internal[k]

            # For the last step we don't have an observation or ground truth (end of experiment)
            if i < test_iter - 1:
                env.set_ground_truth(ground_truth[i + 1])
                obs = observations[i + 1].reshape(obs_shape)

        if done:
            R = 0
        else:
            _, vout = self.model(obs, persistent=True)
            R = float(vout.data)

        for i in range(test_iter-1, -1, -1):

            R = rewards[i] + self.gamma * R
            returns[i] = R

        advantage = returns - value
        advantage_surprise = advantage**2

        return rewards, log_prob, entropy, value, returns, advantage, advantage_surprise, _internal_states
