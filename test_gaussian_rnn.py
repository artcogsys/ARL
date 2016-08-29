# overflow problem encountered

# clean up definition of environment; discrete, nr actions, values, dimensions of state variable, etc etc

# pi should be a 1-d variable. that's why we have the data[0] issue all the time
# this in turn is probably determined by the shape of the observations...
# yes, but 1-d variables is not allowed by layer definitions... closed

# it would actually be more convenient to create mu and sigma2 from the RNN. However,
# (1) needs specialized code in multiple places and also needs redefinition of RNN

# Let's try the latter anyway; might not be too problematic and makes use of continuous model more explicit
# Use discrete flag on RNN? Or create separate RNN model; latter might be cleaner.. return pi = [mu, sigma]
# then just implement the continuous specialisations on the logprob and entropy functions

# we also need proper test code to ensure that we don't break anything!

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import environments
import modelzoo as mz
import agents
from arl import *

## GIVE NETWORK ABILITY TO LEARN ON DATA ACQUIRED BY SUBJECT

###########
# Parameter specification

# learn model
learn = True

# number of processes
nprocs = None

# get file name
name = os.path.splitext(os.path.basename(__file__))[0]

train_iter = 3*10**3 # number training iterations
test_iter = 10**3 # number test iterations

###########
# Environment specification

env = environments.TrackingSine()

###########
# Actor and critic specification

nhidden = 20
model = mz.GaussianRNN(env.ninput, nhidden, env.noutput, covariance = 'fixed')

##########
# Specify agent

agent = agents.A2C(env, model, discrete = False)

###########
# Specify experiment

arl = ARL(agent)

###########
# Learn model

if learn:

    loss, agent = arl.learn(train_iter, nprocs)

    ###########
    # Save model

    agent.save(name)

    if nprocs != 1: # get log loss for one worker
        loss = loss[loss.keys()[0]]

    plt.plot(loss[0], loss[1], 'k')
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

# plot distances between ground truth and action

distances = np.linalg.norm(ground_truth - actions, axis=1)
plt.plot(range(len(distances)), distances, 'k')
plt.xlabel('iteration')
plt.ylabel('distance')
plt.savefig('figures/' + name + '__distances.png')
plt.close()

# plot x position between ground truth and action

#plt.subplot(121)
plt.plot(range(len(ground_truth[:,0])), ground_truth[:,0], 'k')
plt.plot(range(len(actions[:,0])), actions[:,0], 'r')
plt.xlabel('iteration')
plt.ylabel('x position')
# plt.subplot(122)
# plt.plot(range(len(ground_truth[:,1])), ground_truth[:,1], 'k')
# plt.plot(range(len(actions[:,1])), actions[:,1], 'r')
# plt.xlabel('iteration')
# plt.ylabel('y position')
plt.savefig('figures/' + name + '__xy_position.png')
plt.close()

###########
# Analyze run

rewards2, score_function, entropy, value, returns, advantage, advantage_surprise, internal = agent.analyze(ground_truth, observations, actions)

##########
# plot results

# sanity check
plt.plot(range(len(rewards2)), np.cumsum(rewards2), 'k')
plt.xlabel('iteration')
plt.ylabel('cumulative reward')
plt.savefig('figures/' + name + '__reward2.png')
plt.close()

plt.plot(range(len(score_function)), score_function, 'k')
plt.xlabel('iteration')
plt.ylabel('score function')
plt.savefig('figures/' + name + '__score_function.png')
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

plt.plot(range(len(advantage)), advantage, 'k')
plt.xlabel('iteration')
plt.ylabel('advantage')
plt.savefig('figures/' + name + '__advantage.png')
plt.close()

plt.plot(range(len(advantage_surprise)), advantage_surprise, 'k')
plt.xlabel('iteration')
plt.ylabel('surprise')
plt.savefig('figures/' + name + '__surprise.png')
plt.close()

##########
# render results

# agent.render(ground_truth, observations, actions)
