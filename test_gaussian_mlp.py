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

# should we store optimal model? maybe not needed for continuously changing data

# use callback function to store intermediate models?

# gedrag lijkt te zijn: sigma2 gaat naar nul toe; mean blijft dan vast zitten op de plek waar hij zat
# ook al is dit suboptimaal

# to do: score_function code opschonen / afleiden + numerieke problemen vermijden + exploratie forceren

# how to deal with infinite precision actions? Take physical constraints into account

# how to deal with infinite precision observations?

# can we actually deal with negative rewards only?

# DDPG uses deterministic output and motor noise only for exploration; ...

# Bias term essential?

# Taak misschien klassieker kiezen? episodic;
# Check mujoco environment en specificatie van reward daar


import matplotlib.pyplot as plt
import numpy as np
import os
import environments
import modelzoo as mz
import agents
from arl import *
import scipy.stats as sst

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

train_iter = 3*10**3 # number training iterations
test_iter = 10**3 # number test iterations

###########
# Environment specification

env = environments.Constant()

###########
# Actor and critic specification

nhidden = 100
model = mz.GaussianMLP(env.ninput, nhidden, env.noutput, covariance = 'spherical')

##########
# Specify agent

agent = agents.A2C(env, model, discrete = False, file_name = file_name)

###########
# Specify experiment

arl = ARL(agent)

###########
# Custom callback functions

def custom_callback_learning(name, t, losses, action, pi, v, reward):

    if name == 'Process-1' or name == 'Agent':

        if "time" not in custom_callback_learning.__dict__:

            custom_callback_learning.time = []
            custom_callback_learning.sigma2 = []
            custom_callback_learning.mu = []
            custom_callback_learning.v = []

        custom_callback_learning.time.append(t)
        custom_callback_learning.mu.append(pi[0].data[0])
#        custom_callback_learning.sigma2.append(F.exp(pi[1]).data[0])
        custom_callback_learning.sigma2.append(F.softplus(pi[1]).data[0])
        custom_callback_learning.v.append(v.data[0])

        # rough indication of how the loss changes
        print '{0}; {1}; {2:03.5f}'.format(t, name, losses[-1])

        # print 'epoch: {}'.format(model.optimizer.epoch)
        # print 'legend: {}, {}'.format(termcolor.colored('training', 'red'), termcolor.colored('validation', 'green'))
        # print 'loss: {}, {}'.format(termcolor.colored(model.log[('training', 'loss')][-1], 'red'), termcolor.colored(model.log[('validation', 'loss')][-1], 'green'))
        # print 'throughput: {}, {}'.format(termcolor.colored(model.log[('training', 'throughput')][-1], 'red'), termcolor.colored(model.log[('validation', 'throughput')][-1], 'green'))
        #

        plt.clf()
        plt.subplot(311)
        plt.plot(custom_callback_learning.time,custom_callback_learning.mu)
        plt.xlabel('time')
        plt.ylabel('mu')
        plt.subplot(312)
        plt.plot(custom_callback_learning.time,custom_callback_learning.sigma2)
        plt.xlabel('time')
        plt.ylabel('sigma^2')
        plt.subplot(313)
        plt.plot(custom_callback_learning.time,custom_callback_learning.v)
        plt.xlabel('time')
        plt.ylabel('V')
        plt.draw()
#        plt.pause(0.01)

def custom_callback_analyze(file_name, ground_truth, actions, rewards, score_function, entropy, value, returns, advantage, advantage_surprise, _internal_states, pstate, done, observations):

    agent.callback_analyze(file_name, ground_truth, actions, rewards, score_function, entropy, value, returns, advantage, advantage_surprise, _internal_states, pstate, done, observations)

    # plot distances between ground truth and action

    plt.clf()
    distances = np.linalg.norm(ground_truth - actions, axis=1)
    plt.plot(range(len(distances)), distances, 'k')
    plt.xlabel('iteration')
    plt.ylabel('distance')
    plt.savefig('figures/' + file_name + '__distances.png')

    # plot x position between ground truth and action

    # plt.subplot(121)
    plt.clf()
    plt.plot(range(len(ground_truth[:, 0])), ground_truth[:, 0], 'k')
    plt.plot(range(len(actions[:, 0])), actions[:, 0], 'r')
    plt.xlabel('iteration')
    plt.ylabel('x position')
    # plt.subplot(122)
    # plt.plot(range(len(ground_truth[:,1])), ground_truth[:,1], 'k')
    # plt.plot(range(len(actions[:,1])), actions[:,1], 'r')
    # plt.xlabel('iteration')
    # plt.ylabel('y position')
    plt.savefig('figures/' + file_name + '__xy_position.png')

    plt.clf()
    plt.scatter(ground_truth[:, 0], actions[:, 0])
    plt.xlabel('state')
    plt.ylabel('prediction')
    plt.title('R:{}'.format(sst.pearsonr(ground_truth[:, 0],actions[:, 0])[0]))
    plt.savefig('figures/' + file_name + '__scatter.png')


###########
# Learn model

if learn:

    loss, agent = arl.learn(train_iter, nprocs = nprocs) #, callback = custom_callback_learning)

    ###########
    # Save model

    agent.save(file_name)

    if nprocs != 1:  # get log loss for one worker
        loss = loss[loss.keys()[0]]

    plt.clf()
    plt.subplot(211)
    plt.plot(loss[0], np.cumsum(loss[1]), 'k')
    plt.xlabel('iteration')
    plt.ylabel('cumulative loss')
    plt.title('Actor')
    plt.subplot(212)
    plt.plot(loss[0], np.cumsum(loss[2]), 'k')
    plt.xlabel('iteration')
    plt.ylabel('cumulative loss')
    plt.title('Critic')
    plt.savefig('figures/' + file_name + '__loss.png')

    # loss[3] is the reward gained at *each* iteration
    plt.clf()
    plt.plot(range(loss[3].size), np.cumsum(loss[3]), 'k')
    plt.xlabel('iteration')
    plt.ylabel('cumulative reward')
    plt.savefig('figures/' + file_name + '__learning_reward.png')

    # loss[4] is the action gained at *each* iteration
    plt.clf()
    plt.plot(range(loss[4].size), np.cumsum(loss[4]), 'k')
    plt.xlabel('iteration')
    plt.ylabel('cumulative reward')
    plt.savefig('figures/' + file_name + '__learning_action.png')

else:

    # We can also just load an existing model
    agent.load(file_name)

###########
# Run agent

rewards, ground_truth, observations, actions, done = agent.simulate(test_iter)

###########
# Analyze run

agent.analyze(rewards, ground_truth, observations, actions, callback = custom_callback_analyze)

##########
# render results

# agent.render(ground_truth, observations, actions)
