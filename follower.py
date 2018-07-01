'''
Class for line follower bot
'''

import numpy as np
from operator import add, sub
import pygame

class Follower():
    def __init__(self, xinit = 0, yinit = 0, theta = 0, width = 60, length = 80, h_offset = 0, sens_w_offset = 0.25):
        self.width = width
        self.length = length
        self.theta = theta
        self.h_offset = h_offset
        self.sens_w_offset = sens_w_offset

        self.set_points(xinit, yinit, width, length, theta, h_offset)

        self.update_body()

        ### motion parameters
        self.last_update = 0.0

    def set_points(self, xpos, ypos, width, length, theta, h_offset):
        self.theta = theta
        self.center = [xpos, ypos]

        self.width = width
        self.length = length

        self.h_point = [self.center[0] + h_offset * np.cos(self.theta),
                        self.center[1] + h_offset * np.sin(self.theta)]
        self.t_point = [self.center[0] + self.length*np.cos(self.theta + np.pi),
                        self.center[1] + self.length*np.sin(self.theta + np.pi)]

        self.l_point = [self.center[0] + 0.5*self.width*np.cos(self.theta + np.pi/2.0),
                        self.center[1] + 0.5*self.width*np.sin(self.theta + np.pi/2.0)]
        self.r_point = [self.center[0] + 0.5*self.width*np.cos(self.theta - np.pi/2.0),
                        self.center[1] + 0.5*self.width*np.sin(self.theta - np.pi/2.0)]

        self.set_sensor_points()

        self.center = np.array(self.center)
        self.r_point = np.array(self.r_point)
        self.l_point = np.array(self.l_point)
        self.r_sens_point = np.array(self.r_sens_point)
        self.l_sens_point = np.array(self.l_sens_point)
        self.h_point = np.array(self.h_point)
        self.t_point = np.array(self.t_point)

    def set_sensor_points(self):
        self.l_sens_point = [self.h_point[0] + self.sens_w_offset*self.width*np.cos(self.theta + np.pi/2.0), self.h_point[1] + self.sens_w_offset*self.width*np.sin(self.theta + np.pi/2.0)]
        self.r_sens_point = [self.h_point[0] + self.sens_w_offset*self.width*np.cos(self.theta - np.pi/2.0), self.h_point[1] + self.sens_w_offset*self.width*np.sin(self.theta - np.pi/2.0)]

    def get_theta(self):
        direc = np.fromiter(map(sub, self.h_point, self.t_point), dtype=np.float)
        direc = (1.0/np.sqrt(direc[0]**2 + direc[1]**2)) * direc
        return np.arctan2(direc[1], direc[0])

    def update_body(self):
        self.outline = [self.h_point, self.r_sens_point, self.r_point, self.t_point, self.l_point, self.l_sens_point]
        self.lr_bar_points = [self.l_point, self.r_point]
        self.ht_bar_points = [self.h_point, self.t_point]
        dir = np.fromiter(map(sub, self.h_point, self.t_point), dtype=np.float)
        self.theta = np.arctan2(dir[1], dir[0])

    def reset_position(self, xpos = 0, ypos = 0, theta = 0):
        self.theta = theta
        self.set_points(xpos, ypos, self.width, self.length, theta, self.h_offset)
        self.update_body()

    def follower_draw(self, screen):
        BLACK = (0, 0, 0)
        GREEN = (0, 255, 0)
        lr_bar_ints = np.array(self.lr_bar_points).astype(int)
        ht_bar_ints = np.array(self.ht_bar_points).astype(int)
        l_sens_ints = np.array(self.l_sens_point).astype(int)
        r_sens_ints = np.array(self.r_sens_point).astype(int)
        shell_ints = np.array(self.outline).astype(int)
        head_ints = np.array(self.h_point).astype(int)
        self.lr_bar = pygame.draw.lines(screen, BLACK, False,
                                        tuple(lr_bar_ints), 1)
        self.ht_bar = pygame.draw.lines(screen, BLACK, False,
                                        tuple(ht_bar_ints), 1)
        self.shell = pygame.draw.lines(screen, BLACK, True,
                                        tuple(shell_ints), 1)
        self.head = pygame.draw.circle(screen, BLACK, tuple(head_ints), 3)
        self.l_sensor = pygame.draw.circle(screen, BLACK, tuple(l_sens_ints), 3)
        self.r_sensor = pygame.draw.circle(screen, BLACK, tuple(r_sens_ints), 3)

    def translate(self):
        ### manual translate for testing purposes
        self.speedx = 0
        self.speedy = 0
        keystate = pygame.key.get_pressed()
        if keystate[pygame.K_LEFT]:
            self.speedx = -2
        if keystate[pygame.K_RIGHT]:
            self.speedx = 2
        if keystate[pygame.K_UP]:
            self.speedy = -2
        if keystate[pygame.K_DOWN]:
            self.speedy = 2

        self.reset_position(self.center[0] + self.speedx,
                            self.center[1] + self.speedy, self.angle)
        # self.center = [self.center[0] + self.speedx, self.center[1] + self.speedy]
        # self.r_point = [self.r_point[0] + self.speedx, self.r_point[1] + self.speedy]
        # self.l_point = [self.l_point[0] + self.speedx, self.l_point[1] + self.speedy]
        # self.h_point = [self.h_point[0] + self.speedx, self.h_point[1] + self.speedy]
        # self.t_point = [self.t_point[0] + self.speedx, self.t_point[1] + self.speedy]
        #
        # ### generate points of body
        # self.outline = [self.h_point, self.r_point, self.t_point, self.l_point]
        # self.lr_bar_points = [self.l_point, self.r_point]
        # self.ht_bar_points = [self.h_point, self.t_point]
        # self.h_point = (self.h_point[0], self.h_point[1])

    def dual_wheel_move(self, l_speed = 0, r_speed = 0):
        if np.random.rand() < 0.5:
            self.left_wheel_rotate(l_speed)
            self.right_wheel_rotate(r_speed)
        else:
            self.right_wheel_rotate(r_speed)
            self.left_wheel_rotate(l_speed)

    def manual_rotate(self):
        l_speed, r_speed = 0, 0
        keystate = pygame.key.get_pressed()
        if keystate[pygame.K_LEFT]:
            l_speed = 1
        if keystate[pygame.K_RIGHT]:
            r_speed = 1
        if keystate[pygame.K_z]:
            self.reset_position(200,100,np.random.rand() * 2 * np.pi)

        if l_speed == 0 and r_speed == 0:
            return 0

        if np.random.rand() < 0.5:
            self.left_wheel_rotate(l_speed)
            self.right_wheel_rotate(r_speed)
        else:
            self.right_wheel_rotate(r_speed)
            self.left_wheel_rotate(l_speed)

    def left_wheel_rotate(self, l_speed):
        if l_speed == 0:
            return 0

        ### subtract h ant t points to get direction
        direc = np.fromiter(map(sub, self.h_point, self.t_point), dtype=np.float)
        direc = (1.0/np.sqrt(direc[0]**2 + direc[1]**2)) * direc

        ### translate l  points
        self.l_point = [self.l_point[0] + l_speed * direc[0],
                        self.l_point[1] + l_speed * direc[1]]

        ### set center to be w/2 away from r point toward new l point
        w_direc = np.fromiter(map(sub, self.l_point, self.r_point), dtype=np.float)
        w_direc = w_direc / np.sqrt(w_direc[0]**2 + w_direc[1]**2)
        w_diff = 0.5* self.width * w_direc
        self.center = np.fromiter(map(add, self.r_point, w_diff), dtype=np.float)

        ### reposition new l point so that it is w/2 away from new center
        l_diff = self.width * w_direc
        self.l_point = np.fromiter(map(add, self.r_point, l_diff), dtype=np.float)

        ### get new direction of lr bar
        perp = np.fromiter(map(sub, self.r_point, self.l_point), dtype=np.float)
        perp /= np.sqrt(perp[0]**2 + perp[1]**2)

        ### get new direction of ht bar
        direc = [-perp[1], perp[0]]
        direc = np.array(direc)

        ### set new h and t points
        self.h_point = [self.center[0] + self.h_offset * np.cos(self.theta),
                        self.center[1] + self.h_offset * np.sin(self.theta)]
        self.t_point = np.fromiter(map(add, self.center, -self.length * direc), dtype=np.float)

        self.theta = self.get_theta()
        self.set_sensor_points()

        ### generate points of body
        self.update_body()

    def right_wheel_rotate(self, r_speed):
        if r_speed == 0:
            return 0

        ### subtract h ant t points to get direction
        direc = np.fromiter(map(sub, self.h_point, self.t_point), dtype=np.float)
        direc = (1.0/np.sqrt(direc[0]**2 + direc[1]**2)) * direc

        ### translate l  points
        self.r_point = [self.r_point[0] + r_speed * direc[0],
                        self.r_point[1] + r_speed * direc[1]]

        ### set center to be w/2 away from l point toward new r point
        w_direc = np.fromiter(map(sub, self.r_point, self.l_point), dtype=np.float)
        w_direc = w_direc / np.sqrt(w_direc[0]**2 + w_direc[1]**2)
        w_diff = 0.5* self.width * w_direc
        self.center = np.fromiter(map(add, self.l_point, w_diff), dtype=np.float)

        ### reposition new r point so that it is w/2 away from new center
        r_diff = self.width * w_direc
        self.r_point = np.fromiter(map(add, self.l_point, r_diff), dtype=np.float)

        ### get new direction of lr bar
        perp = np.fromiter(map(sub, self.r_point, self.l_point), dtype=np.float)
        perp /= np.sqrt(perp[0]**2 + perp[1]**2)

        ### get new direction of ht bar
        direc = [-perp[1], perp[0]]
        direc = np.array(direc)

        ### set new h and t points
        self.h_point = [self.center[0] + self.h_offset * np.cos(self.theta),
                        self.center[1] + self.h_offset * np.sin(self.theta)]
        self.t_point = np.fromiter(map(add, self.center, -self.length * direc), dtype=np.float)

        self.theta = self.get_theta()
        self.set_sensor_points()

        ### generate points of body
        self.update_body()

    def wrap_vertical(self, size):
        diff = 0.0
        new_y = 0.0
        if self.center[1] < 0:
            diff = self.center[1] - 0
            new_y = grad.border_width - diff
        elif self.center[1] > size[1]:
            diff = self.center[1] - size[1]
            new_y = grad.border_width + diff
        else:
            return 0
        self.reset_position(self.center[0], new_y, self.theta)
        self.update_body()

    def wrap_horizontal(self, size):
        diff = 0.0
        new_y = 0.0
        if self.center[1] < 0:
            diff = self.center[1] - 0
            new_y = grad.border_width - diff
        elif self.center[1] > size[1]:
            diff = self.center[1] - size[1]
            new_y = grad.border_width + diff
        else:
            return 0
        self.reset_position(self.center[0], new_y, self.theta)
        self.update_body()


    ### observation returns the gradient values at the points specified
    ### relies on grad_xy_rgb function defined in the gradient class
    def observation(self, screen, points):
        obs = []
        for p in points:
            ptint = (int(np.rint(p[0])), int(np.rint(p[1])))
            obs.append(screen.get_at(ptint))
        obs = np.array(obs)
        obs = obs.astype(int)
        return obs

    # def check_out_of_bounds(self, size):
    #     check = 0
    #     run = True
    #     for p in self.outline:
    #         check += 1 if p[0] > size[0]
    #         check += 1 if p[1] > size[1]
    #         check += 1 if p[0] < 0
    #         check += 1 if p[1] < 0
    #     if check > 0:
    #         run = False
    #     return run

    # ### defines the reward for the current state/environment
    # def reward(self, gradient):
    #     return 0

    def update(self):
        self.last_update += pygame.time.get_ticks()
