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

import matplotlib.pyplot as plt
import numpy as np
import os
import environments
import modelzoo as mz
import agents
from arl import *

# set interactive mode
plt.ion()

## GIVE NETWORK ABILITY TO LEARN ON DATA ACQUIRED BY SUBJECT

###########
# Parameter specification

# learn model
learn = True

# number of processes
nprocs = None

# get file name
file_name = os.path.splitext(os.path.basename(__file__))[0]

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

agent = agents.A2C(env, model, discrete = False, file_name = file_name)

###########
# Specify experiment

arl = ARL(agent)

###########
# Learn model

if learn:

    loss, agent = arl.learn(train_iter, nprocs = nprocs, callback = None)

    ###########
    # Save model

    agent.save(file_name)

    if nprocs != 1:  # get log loss for one worker
        loss = loss[loss.keys()[0]]

    plt.clf()
    plt.subplot(211)
    plt.plot(loss[0], loss[1], 'k')
    plt.xlabel('iteration')
    plt.ylabel('actor loss')
    plt.subplot(212)
    plt.plot(loss[0], loss[2], 'k')
    plt.xlabel('iteration')
    plt.ylabel('critic loss')
    plt.savefig('figures/' + file_name + '__loss.png')


else:

    # We can also just load an existing model
    agent.load(file_name)

###########
# Run agent

rewards, ground_truth, observations, actions, done = agent.simulate(test_iter)

###########
# Analyze run

agent.analyze(rewards, ground_truth, observations, actions, callback = None)

##########
# render results

# agent.render(ground_truth, observations, actions)
