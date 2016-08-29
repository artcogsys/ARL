# check formulation gaussian_nll and write in document
# - nll or + nll ?
# simplify problem as much as possible (1D)
# check numerical problems
# double check if original code is not broken now => ok, benchmark test required!
# reward was positive for more distance. now fixed to negative reward. other problems in standard formulation?
# integer representation of actions fixed
# interactie tussen negative reward, negative loglik, input to policy gradient, optimizer etc? (flipping of sign somewhere)
# does something go wrong only after many iterations??? instabilities?
# is particle tracking een onmogelijk probleem? what about sine wave?

# check numerical outputs (instabilities? few learning cycles seems fine)
# can the problem be learn using a standard MLP? That should be the gold standard
# Not enough unique repetition steps for sine wave problem /100 ??? (10 * the whole wave is sampled in sequence

# moving to even simpler by breaking temporal dependencies: random number is underlying state
# hmm. interactie met advantage of value computation???
# merk op dat we genoeg hidden nodes nodig hebben om de actie te voorspellen, de variance, en de value
# haalbaar met 1 netwerk? denk t wel
# denk ook aan interactie met gekozen variance; initalisatie ver weg kan tot extreem kleine getallen leiden...

# waarom perfect predictie en offset? komt dat niet tot uitdrukking in de reward? bv 1 / norm geeft een ander effect

# we hebben sowieso een probleem als we de variance gaan schatten want die gaat naar nul in geval van perfect prediction

# af laten nemen van fixed variance over time zou ook nog kunnen werken; check ook bestaande implementaties/papers/etc
# zie eventueel werk van pieter abbeel e.d.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import environments
import modelzoo as mz
import agents
from arl import *

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

env = environments.RandomSample()

###########
# Actor and critic specification

nhidden = 20
model = mz.GaussianMLP(env.ninput, nhidden, env.noutput, covariance = 'fixed')

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

plt.scatter(ground_truth[:,0], actions[:,0])
plt.xlabel('state')
plt.ylabel('prediction')
plt.savefig('figures/' + name + '__scatter.png')
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
plt.ylabel('score_function')
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
