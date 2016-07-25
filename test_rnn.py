import numpy as np
import os
import environments
import modelzoo as mz
import agents
import matplotlib.pyplot as plt
from arl import *

## GIVE NETWORK ABILITY TO LEARN ON DATA ACQUIRED BY SUBJECT

###########
# Parameter specification

# learn model
learn = True

# get file name
name = os.path.splitext(os.path.basename(__file__))[0]

train_iter = 3*10**4 # number training iterations
test_iter = 10**3 # number test iterations

###########
# Environment specification

# n = 5
# p = 0.8
# env = environments.Foo(n,p)

odds = [0.25, 0.5, 2, 4]
#odds = [0.5, 0.9, 0.9, 1.1, 1.1, 2]
#odds = [0.5, 2]
env = environments.ProbabilisticCategorization(odds)

###########
# Actor and critic specification

nhidden = 20
model = mz.RNN(env.ninput, nhidden, env.noutput)

##########
# Specify agent

agent = agents.A2C(env, model)

###########
# Specify experiment

arl = ARL(agent)

###########
# Learn model

if learn:

    losses, agent = arl.learn(train_iter)

    ###########
    # Save model

    agent.save(name)

    ###########
    # plot log loss for one worker
    ts = losses[losses.keys()[0]][0]
    loss = losses[losses.keys()[0]][1]

    plt.plot(ts, loss, 'k')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.savefig('figures/' + name + '__loss.png')
    plt.close()

else:

    # We can also just load an existing model
    agent.load(name)

###########
# Run agent

rewards, ground_truth, observations, actions, done = agent.simulate(test_iter)

##########
# visualize results

# plot rewards

plt.plot(range(len(rewards)), np.cumsum(rewards), 'k')
plt.xlabel('iteration')
plt.ylabel('cumulative reward')
plt.savefig('figures/' + name + '__reward.png')
plt.close()

###########
# Analyze run

rewards2, log_prob, entropy, value, returns, hidden = agent.analyze(ground_truth, observations, actions)

##########
# visualize results

# sanity check
plt.plot(range(len(rewards2)), np.cumsum(rewards2), 'k')
plt.xlabel('iteration')
plt.ylabel('cumulative reward')
plt.savefig('figures/' + name + '__reward2.png')
plt.close()

plt.plot(range(len(log_prob)), log_prob, 'k')
plt.xlabel('iteration')
plt.ylabel('log probability')
plt.savefig('figures/' + name + '__log_prob.png')
plt.close()

plt.plot(range(len(entropy)), entropy, 'k')
plt.xlabel('iteration')
plt.ylabel('entropy')
plt.savefig('figures/' + name + '__entropy.png')
plt.close()

plt.plot(range(len(value)), value, 'k')
plt.xlabel('iteration')
plt.ylabel('value')
plt.savefig('figures/' + name + '__value.png')
plt.close()

plt.plot(range(len(returns)), returns, 'k')
plt.xlabel('iteration')
plt.ylabel('returns')
plt.savefig('figures/' + name + '__returns.png')
plt.close()