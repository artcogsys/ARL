import numpy as np
import sys
import matplotlib.pyplot as plt

###
# Base class for an environment

class Environment(object):

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        pass

    def get_ground_truth(self):
        pass

    def set_ground_truth(self, ground_truth):
        pass

    def generate(self):
        pass


###
# Specific environments

class Foo(Environment):
    """
    Very simple environment for testing fully observed models. The actor gets a reward when it correctly decides
    on the ground truth. Ground truth 0/1 determines probabilistically the number of 0s or 1s on the output
    """

    def __init__(self, n, p = 0.8):
        """

        Args:
            n: number of inputs
            p: probability of emitting the right sensation at the input
        """

        super(Foo, self).__init__()

        self.ninput = n
        self.p = p
        self.noutput = 2

        self.reset()

    def reset(self):

        self.state = np.random.randint(0, 2)

        p = np.array([1 - self.p, self.p])

        if self.state == 0:
            p = 1 - p

        obs = np.random.choice(2, [1, self.ninput], True, p)

        return obs.astype(np.float32)

    def step(self, action):

        # reward is +1 or -1
        reward = 2 * int(action == self.state) - 1

        obs = self.reset()
        done = True

        return obs, reward, done

    def get_ground_truth(self):
        """
        Returns: ground truth state of the environment
        """

        return self.state

    def set_ground_truth(self, ground_truth):
        """
        :param: ground_truth : sets ground truth state of the environment
        """

        self.state = ground_truth

class ProbabilisticCategorization(Environment):
    """

    Formerly known as PerceptualDecision task

    Let odds be a vector determining the odds ratio for emitting a certain symbol x = i given state k = j:

        odds = [ P(x = 0 | k = 1) / P(x = 0 | k = 0) ... P(x = n | k = 1) / P(x = n | k = 0) ]

    We define

        p = odds / sum(odds)
        q = (1/odds) / sum(1/odds)

    Let P(x = i | p, q, k) define the probability that the emitted symbol is i given that we have probability vector p and q and
    the true state can be either k = 0 or k = 1. Then

        P(x = i | p, k) = p^k * q^(k-1)

    Note: vector of zeros indicates absence of evidence (starting state)

    Volatility could be implemented by defining multiple such distributions and some volatility parameter
    see also work Behrens and auditory stuff by Stephan at HBM 2016


    """

    def __init__(self, odds = [0.25, 0.5, 2, 4]):
        """

        :param: odds : determines odds ratio

        """

        super(ProbabilisticCategorization, self).__init__()

        self.odds = np.array(odds)

        self.p = self.odds/float(np.sum(self.odds))
        self.q = (1.0/self.odds)/float(np.sum(1.0/self.odds))

        self.ninput = len(self.p)
        self.noutput = 3

        self.rewards = [-1, 10, -100]

        # normalize rewards
        self.rewards = np.array(self.rewards) / float(np.sum(np.array(self.rewards)))

        self.reset()

    def get_ground_truth(self):
        """
        Returns: ground truth state of the environment
        """

        return self.state


    def set_ground_truth(self, ground_truth):
        """
        :param: ground_truth : sets ground truth state of the environment
        """

        self.state = ground_truth


    def reset(self):

        self.state = np.random.randint(1, 3)  # 1 = left, 2 = right

        return np.zeros([1, self.ninput], dtype='float32')


    def render(self):

        print self.state


    def step(self, action):

        if action == 0:  # wait to get new evidence

            reward = self.rewards[0]

            # choose piece of evidence
            if self.state == 1:
                evidence = np.random.choice(self.ninput, p = self.p)
            else:
                evidence = np.random.choice(self.ninput, p = self.q)

            obs = np.zeros([1, self.ninput], dtype='float32')
            obs[0, evidence] = 1

            done = False

        else:  # left or right was chosen

            if action == self.state:
                reward = self.rewards[1]
            else:
                reward = self.rewards[2]

            done = True

            obs = self.reset()

        return obs, reward, done


class WeatherPrediction(Environment):
    """

    Classical weather prediction task

    See probabilistic categorization task of Yang and Shadlen, 2007

    The weight of a sequence of symbols defines the evidence for / against a target

    To be implemented

    """

    pass

class RandomDotMotion(Environment):
    """
    Prototype for the Random Dot Motion environment that should be replaced by the
    actually chosen experimental framework. 
    """

    def __init__(self, n_dots = 500, speed = 2, coherence = 0.5, pixwidth=400, border = True, flatten = False, circular = True):
        """
        move_dots() will update the current observation (obs_screen).

        3 actions: wait, choose_left, choose_right
        observations: [pixwidth x pixwidth]

        Input:
        n_dots:      number of dots
        speed:       dot movement speed in pix per update / refresh
        coherence:   coherence level; if it is a vector then a random values will be chosen at each iteration
        pixwidth:    Width (and height) of stimulus
        border:      Will add 10-15% frame border (i.e. region with no dots) if true. 
        flatten:     Whether or not to flatten the output observations
        circular:    Whether random dot field should be circular. Will cover square if False. 
        
        
        Get environment state:
        Percentage of coherent dots: SimpleMotionCoherence.get_coherence()
        
        """

        super(RandomDotMotion, self).__init__()

        if isinstance(coherence,list):
            self.coherence = coherence
        else:
            self.coherence = [coherence]

        self.n_dots = n_dots

        self.pixwidth = pixwidth
        
        self.circular = circular
        
        if border: 
            self.circle_radius  = (self.pixwidth - int(0.15*self.pixwidth)) / 2.  # integer rounding / off
        else:
            self.circle_radius  = (self.pixwidth) / 2.
            
        self.central_pos = self.pixwidth / 2.

        self.lifetime_dots = 10    # frames

        self.speed = speed   # pix / frame

        self.display_refresh_rate = 60

        if self.pixwidth**2 < self.n_dots:
            sys.exit("Please use a valid number of dots.")

        self.ninput = [self.pixwidth, self.pixwidth]
        self.noutput = 3

        self.obs_screen = np.zeros(self.ninput, dtype='float32')
        self.dots_pos_xy = np.zeros([2, self.n_dots])

        # Can be used to apply non-convolutional models to this environment
        self.flatten = flatten

        self.rewards = [-1, 10, -100]

        # normalize rewards
        self.rewards = np.array(self.rewards) / float(np.sum(np.array(self.rewards)))

        # Now initialize the dots with above parameters:
        self.reset()

        # Figure handle
        self.fig = None

        # print "State: ", self.state, " | Motion coherence:", self.get_coherence(),"%"

    def get_ground_truth(self):
        """
        Returns: ground truth state of the environment
        """

        return self.state

    def set_ground_truth(self, ground_truth):
        """
        :param: ground_truth : sets ground truth state of the environment
        """

        self.state = ground_truth

    def get_coherence(self):

        return np.rint(self.p_coherent * 100.0).astype(int)


    def get_screen_with_dots(self):

        self.obs_screen = np.zeros([self.pixwidth, self.pixwidth], dtype='float32')
        #self.obs_screen.fill(0)  # fill screen with blanks

        for dot_i in xrange(self.n_dots):
            #x,y = np.rint( self.dots_pos_xy[:,dot_i] ).astype(int)  # round position to nearest integer
            x,y = int(self.dots_pos_xy[0,dot_i]), int(self.dots_pos_xy[1,dot_i])
            self.obs_screen[y,x] = 1.0

        # make zero centered
        self.obs_screen -= 0.5

        if self.flatten:
            return self.obs_screen.reshape([1,self.obs_screen.size])
        else:
            # np.expand_dims
            return self.obs_screen.reshape(np.hstack([1,self.ninput]))


    def init_dot(self):
        # get random distance from center and random position angle: 
        angle = 2. * np.pi * np.random.rand()
        dist_from_center =  self.circle_radius * np.sqrt(np.random.rand())
        x = np.floor(self.central_pos + dist_from_center * np.sin(angle))
        y = np.floor(self.central_pos + dist_from_center * np.cos(angle))  # init at exact position
        return [x,y]


    def reset(self):

        self.p_coherent = np.random.choice(self.coherence)
        self.n_coh_dots = self.p_coherent * self.n_dots

        self.state = np.random.randint(1,3)  # Motion direction: 1 = left, 2 = right
        self.motiondirection = (-0.5 if self.state == 1 else 0.5) * np.pi   # motion direction in rad

        # Create the dots: 
        self.dots_pos_xy = np.zeros([2, self.n_dots])        

        # Initialize the dots: 
        for dot_i in xrange(self.n_dots):

            # place dots: 
            self.dots_pos_xy[:,dot_i] = self.init_dot()

        # Return current observation (i.e. initial dots): 
        return self.get_screen_with_dots()


    def move_dots(self):
        for dot_i in xrange(self.n_dots): 
            if dot_i < self.n_coh_dots:   # this is a coherent dot
                motion_angle = self.motiondirection
            else:    # this is a randomly moving dot
                motion_angle = 2. * np.pi * np.random.rand()
                
            # update dot position: 
            self.dots_pos_xy[:,dot_i] = [ self.dots_pos_xy[0,dot_i] + np.sin(motion_angle) * self.speed,
                                          self.dots_pos_xy[1,dot_i] + np.cos(motion_angle) * self.speed]

            if self.circular:
                # put coherent dot back inside if it moves outside circular visible area: 
                is_outside = ( ( (self.central_pos - self.dots_pos_xy[0,dot_i])**2 + 
                                 (self.central_pos - self.dots_pos_xy[1,dot_i])**2 ) > self.circle_radius**2 )
            else:
                is_outside = ( not (0.0 <= self.dots_pos_xy[0,dot_i] < self.pixwidth ) or
                               not (0.0 <= self.dots_pos_xy[0,dot_i] < self.pixwidth ) )

            if is_outside: 
                self.dots_pos_xy[:,dot_i] = self.init_dot()
    
        return self.get_screen_with_dots()


    def render(self):
        # Rendering only works with pylab==tk:
        # e.g. ipython wait_motion_coherence.py --pylab=tk

        if not self.fig:
           self.fig = plt.figure()

        plt.clf()
        plt.imshow(self.obs_screen, cmap='Greys_r', interpolation='nearest')
        self.fig.canvas.draw()

        # for i in xrange(self.ninput[0]):
        #
        #     for j in xrange(self.ninput[1]):
        #
        #         print self.obs_screen.T[i,j],
        #         print ' ',
        #
        #     print '\n'
        #
        # raw_input()


    def step(self, action):
        # if action: check whether correct
        # if not: advance one frame

        if action == 0:   # wait to get new evidence
            reward = self.rewards[0]

            done = False

            return self.move_dots(), reward, done

        else:    # left or right was chosen

            if action == self.state:
                reward = self.rewards[1]
            else:
                reward = self.rewards[2]

            done = True

            return self.reset(), reward, done


