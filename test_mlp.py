import numpy as np
import os
import environments
import modelzoo as mz
import agents
import matplotlib.pyplot as plt
from arl import *

# action probabilities go to 1
# entropy should encourage exploration
# read standard advantage actor critic
# check website silver
# seems it has no time to figure out right policy

###########
# Parameter specification

# get file name
name = os.path.splitext(os.path.basename(__file__))[0]

train_iter = 5*10**3 # number training iterations
test_iter = 10**3 # number test iterations

###########
# Environment specification

# n = 5
# p = 0.8
# env = environments.Foo(n,p)

odds = [0.25, 0.5, 2, 4]
env = environments.ProbabilisticCategorization(odds)

###########
# Actor and critic specification

nhidden = 10
nframes = 5
model = mz.MLP(env.ninput, nhidden, env.noutput, nframes=nframes)

###########
# Specify agent

agent = agents.A2C(env, model)

###########
# Specify experiment

arl = ARL(agent)

###########
# Learn model

loss = arl.learn(train_iter)

###########
# plot log loss

plt.plot(np.arange(len(loss)), loss, 'k')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.savefig('figures/' + name + '__loss.png')
plt.close()

###########
# Run agent

# We could load a model directly from disk
# from chainer import serializers
# arl.agent.model = serializers.load_npz('models/TestARL_MLP.model')

# ground_truth, observations, actions, rewards, qvalues, terminal, internal = agent.run(test_iter)

rewards = arl.run(test_iter)

##########
# visualize results

# plot rewards

plt.plot(range(len(rewards)), np.cumsum(rewards), 'k')
plt.xlabel('iteration')
plt.ylabel('cumulative reward')
plt.savefig('figures/' + name + '__reward.png')
plt.close()

