import numpy as np
import os
import environments
import modelzoo as mz
import agents
import matplotlib.pyplot as plt
from arl import *

###########
# Parameter specification

# get file name
name = os.path.splitext(os.path.basename(__file__))[0]

train_iter = 10**4 # number training iterations
test_iter = 10**3 # number test iterations

###########
# Environment specification

ndots = 1
pixwidth = 3
coherence = 1 # [0.1, 0.9]  # will be selected from with random.choice()

env = environments.RandomDotMotion(n_dots = ndots, speed = 1, coherence = coherence, pixwidth=pixwidth, flatten = False,
                                   circular = False)

###########
# Actor and critic specification

nhidden = 20
nframes = 3
model = mz.CNN(env.ninput, nhidden, env.noutput, nframes = nframes)


##########
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

rewards = arl.run(test_iter)

##########
# visualize results

# plot rewards

plt.plot(range(len(rewards)), np.cumsum(rewards), 'k')
plt.xlabel('iteration')
plt.ylabel('cumulative reward')
plt.savefig('figures/' + name + '__reward.png')
plt.close()

