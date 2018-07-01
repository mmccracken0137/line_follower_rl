'''
Policy-gradient line-following learner
***MUST*** be run with pythonw in order to properly focus the visualization window.
Adapted from https://github.com/awjuliani/DeepRL-Agents/blob/master/Policy-Network.ipynb
Also see Karpathy's blog: http://karpathy.github.io/2016/05/31/rl/
ALSO see https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
'''

import pygame
import numpy as np
import os
import sys
import tensorflow as tf
import pickle

from follower import *
from closed_path import *

'''
Set up a MLP with a single hidden layer.
Actions will be simplest possible: each wheel can move forward (+1) independently.
'''

# hyperparameters
n_hidden = 50 # number of nodes in the hidden layer
batch_size = 20 # run this many episodes before doing a parameter update
learning_rate = 1e-3
gamma = 0.99 # discount factor
decay_rate = 0.99 # decay factor for RMSprop leaky sum of grad^2 ???
kindness = 0.2 # frequency that non-highest prob action will not be chosen (non-greedy choice)  ALERT ALERT this may not work for multi-output nets (i.e. >2 possible actions)...
random_position_reset = True # reset the position of the rover randomly for each episode

render = False # run visualization for each episode?
resume = False # resume from previous trainng session?
# render = True # run visualization for each episode?
# resume = True # resume from previous trainng session?

write_inits = False # write initial values to file???
save_inits = None
if write_inits:
    save_inits = open('inits.txt', 'w')

write_fom = False # write figure of merit to file.  show training progress...
save_fom = None
if write_fom:
    save_fom = open('fom.txt', 'w')

# number of episodes to run
max_epis = 100000
max_steps = 1500

# model initialization
n_inputs = 3 # take only one color from sensor input -- distinguishes black/white
# n_actions = 2

# if we're going to pick up from a previous training session, load the model. otherwise initialize!
if resume:
    save_file = sys.argv[1]
    print('\nloading model from %s...\n' % save_file)
    model = pickle.load(open(save_file, 'rb'))
else:
    print('\ninitializing fresh model...\n')
    model = {}
    # will try to multiply in the following order: h = W1.x, y = W2.h.  This dictates the dimension of the weights matrices.
    model['W1'] = np.random.randn(n_hidden, n_inputs) / np.sqrt(n_inputs) # Xavier init
    model['W2'] = np.random.randn(n_hidden) / np.sqrt(n_hidden)

grad_buffer = {k : np.zeros_like(v) for k, v in model.items() } #update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k, v in model.items() } # rmsprop memory

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    ''' applies softmax to array of values '''
    probs = []
    sum = 0
    max_x = np.amax(x)
    for i in range(len(x)):
        ex = np.exp(x[i] - max_x)
        probs.append(ex)
        sum += ex
    probs = np.array(probs)
    probs /= sum
    return probs

def prepro(I):
    out = []
    for s in I:
        out.append(s[1])
    out = np.array(out).flatten()
    #out = np.ndarray.flatten(out)
    out = out / 255.0
    return out.astype(np.float)

def discount_rewards(r):
    ''' take 1d array of float rewards and compute discounted reward '''
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.dot(model['W1'], x) # multiply W1 by input layer
    h[h<0] = 0 # ReLU, baby!!!
    logp = np.dot(model['W2'], h) # multiply W2 and hidden layer
    p = sigmoid(logp)
    return p, h # return probs array and hidden state

def policy_backward(epx, eph, epdlogp):
    ''' backward pass. (eph is array of intermediate hidden states). see etienne87 '''
    # from Karpathy...
    dW2 = np.dot(eph.T, epdlogp).ravel()  # TKTK from stackexchange!!!
    dh = np.outer(epdlogp, model['W2']) # np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0 # backprop ReLU nonlinearity
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}

def follower_action(a_index):
    '''
    only two possible actions for this script: either wheel moves forward
    '''
    a = np.zeros(2)
    if a_index == 0:
        a[0] = 1
    elif a_index == 1:
        a[1] = 1
    return a

def follower_reward(ob, act, stp):
    ''' defines the reward function for the follower. observation is a list of color tuples. action is a pair of wheel motions. '''
    #print(ob)
    reward, penalty = 0.0, 0.0
    x = prepro(ob)
    neg = np.zeros(len(x))
    for i in range(len(x)):
        neg[i] = abs(x[i] - 1)

    sum = neg.sum()
    if sum > 0.0:
        reward += 1e-2 * sum
    else:
        penalty += -1e-2

    return reward, penalty

def check_running(ob, act, stp, rew_sum, pen_sum):
    '''
    conditions under which to end the episode...
    '''

    run = True
    #end episode if positive or negative reward is achieved
    if pen_sum <= -1:
        run = False

    return run

# these objs will hold quantities for each episode
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
penalty_sum = 0
episode_number = 0

# set up visualization
size = (700, 700)
screen_center = [size[0]/2, size[1]/2]
screen = pygame.display.set_mode(size)
pygame.display.set_caption("closed track")

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

surf = pygame.Surface(size)
surf.fill(WHITE)

path = ClosedPath(npts = 500, bias = 200, order = 4, line_width = 3, center = [size[0]/2, size[1]/2], amp = 140)
path.draw_path(surf, BLACK)

if render:
    pygame.display.flip()

screen.blit(surf, (0,0))

### initialize foll
initt = np.random.rand() * 2 * np.pi
initx = path.get_x(initt)
inity = path.get_y(initt)
init_phi = path.tangent_angle(initt)
foll = Follower(initx, inity, init_phi, width = 30, length = 30, h_offset = 10, sens_w_offset = 0.3)
foll.reset_position(initx, inity, init_phi)

clock = pygame.time.Clock()

epi = 1

while epi < max_epis:
    epi += 1
    steps = 0
    if random_position_reset:
        initt = np.random.rand() * 2 * np.pi
    else:
        initt = 0

    initx = path.get_x(initt)
    inity = path.get_y(initt)
    init_phi = path.tangent_angle(initt)
    foll.reset_position(initx, inity, init_phi)

    if write_inits:
        save_inits.write('%f\t%f\t%f\t' % (foll.center[0], foll.center[1], foll.theta))

    running = True

    while running:
        if render:
            clock.tick(60)

        # shouldn't need this here...
        surf.fill(WHITE)
        path.draw_path(surf, BLACK)
        screen.blit(surf, (0,0))

        # get follower obs and preprocess
        sens_pts = [foll.l_sens_point, foll.h_point, foll.r_sens_point]
        x = prepro(foll.observation(surf, sens_pts))
        #print(x)

        # forward policy network and get probabilities for actions
        aprob, h = policy_forward(x)

        a_ind, y = 0, 0
        if np.random.uniform() < aprob:
            a_ind = 0
            y = 1
        else:
            a_ind = 1
            y = 0

        xs.append(x) # observation
        hs.append(h) # hidden state

        dlogps.append(y - aprob)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False # Flag that we are done so we exit this loop

        # --- Game logic should go here ---> update
        # update and observe, step rover.  Karpathy line 90-ish
        action = follower_action(a_ind)
        foll.dual_wheel_move(action[0], action[1])
        foll.update()

        # --- Drawing code should go here ---> draw
        # still need to remap path to surf for sensors to work
        surf.fill(WHITE)
        path.draw_path(surf, BLACK)
        foll.follower_draw(surf)
        screen.blit(surf, (0,0))
        if render:
            #screen.fill((255,255,255))
            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

        steps += 1
        if steps % 1000 == 0:
            print(steps, ' steps')
        #

        # shouldn't need this here...
        surf.fill(WHITE)
        path.draw_path(surf, BLACK)
        screen.blit(surf, (0,0))
        sens_pts = [foll.l_sens_point, foll.h_point, foll.r_sens_point]
        obs = foll.observation(surf, sens_pts)

        f_theta = path.get_theta_from_xy(foll.center)

        reward, penalty = follower_reward(obs, action, steps)
        reward_sum += reward
        penalty_sum += penalty
        drs.append(reward + penalty)

        running = check_running(obs, action, steps, reward_sum, penalty_sum)

        if running == False:
            if write_fom:
                save_fom.write('%d\t%d\t%d\n' % (init_goal_dist, steps, reward))
            if write_inits:
                save_inits.write('%f\t%f\t%f\n' % (foll.center[0], foll.center[1], foll.theta))

            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs, hs, dlogps, drs = [], [], [], [] # reset array memory

            # compute discounted reward
            discounted_epr = discount_rewards(epr)
            # standardize rewards to be unit norm
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            # modulate the gradient with the advantage !!!
            epdlogp *= discounted_epr
            # print(epx.shape, eph.shape, epdlogp.shape, discounted_epr.shape)
            grad = policy_backward(epx, eph, epdlogp)
            for k in model:
                #print(grad_buffer[k].shape, grad[k].shape)
                grad_buffer[k] += grad[k] # accumulate grad over batch

            # now we perform the rmsprop parameter update every time a batch is finished
            if epi % batch_size == 0:
                print('\nbatch ended --> back prop!\n')
                for k, v in model.items():
                    g = grad_buffer[k] # gradient
                    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5) # ???
                    grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

            # book-keeping
            if running_reward is None:
                running_reward = reward_sum + penalty_sum
            else:
                running_reward = running_reward * 0.99 + (reward_sum + penalty_sum) * 0.01

            batch_mean = running_reward
            print('episode %d took %d steps.\tepisode reward total was %.2f.\trunning mean: %.4f' % (epi, steps, reward_sum + penalty_sum, running_reward))

            # pickle every 100 episodes
            if epi % 100 == 0: pickle.dump(model, open('save_' + str(n_hidden) + 'h.p', 'wb'))

            reward_sum, penalty_sum = 0, 0
            obs = None


#Once we have exited the main program loop we can stop the game engine:
pygame.quit()
if write_inits:
    save_inits.close()
if write_fom:
    save_fom.close()
