import numpy as np
from chainer import optimizers, Variable
import chainer.functions as F
from asynchronous import RMSpropAsync

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

        Returns: loss

        """

        # reset the environment and get observation
        obs = self.environment.reset()

        losses = []

        # create trajectory of learning rates
        learning_rates = np.linspace(self.optimizer.lr, 0, niter, False)

        # outer training loop
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

                # get output of actor-critic model for this observation
                pi, v = self.model(obs)

                # generate action according to policy
                p = F.softmax(pi)
                action = np.random.choice(self.noutput, None, True, p.data[0])
                # print p.data

                # store log probability of the action which was selected
                logp = F.log_softmax(pi)
                self.past_action_log_prob[idx] = F.select_item(logp, Variable(np.asarray([action], dtype=np.int32)))

                # perform action via actor and receive new observations and reward
                obs, reward, done = self.environment.step(action)

                # No reward clipping; will lead to random guessing as best option for our environment
                # reward = np.clip(reward, -1, 1)

                # store reward and value
                self.past_rewards[idx] = reward
                self.past_values[idx] = v

                # compute entropy
                self.past_action_entropy[idx] = - F.sum(p * logp, axis=1)

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

        return losses

    def act(self, obs):
        """

        Advantage actor-critic is always on-policy

        :param obs: sensory input
        :return: action
        """

        # get output of actor model for this observation
        pi, _ = self.model(obs)

        # generate action according to policy
        p = F.softmax(pi).data[0]

        # return action chosen according to stochastic policy
        return np.random.choice(self.noutput, None, True, p)
