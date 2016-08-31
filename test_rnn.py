import matplotlib.pyplot as plt
import numpy as np
import os
import environments
import modelzoo as mz
import agents
from arl import *

# set interactive mode
plt.ion()

###########
# Parameter specification

# learn model
learn = True

# number of processes
nprocs = None

# get file name
file_name = os.path.splitext(os.path.basename(__file__))[0]

train_iter = 3*10**4# number training iterations
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

agent = agents.A2C(env, model, file_name = file_name)

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

    if nprocs != 1: # get log loss for one worker
        loss = loss[loss.keys()[0]]

    plt.clf()
    plt.plot(loss[0], loss[1], 'k')
    plt.xlabel('iteration')
    plt.ylabel('loss')
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
