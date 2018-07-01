import pygame
import numpy as np
import matplotlib.pyplot as plt

class ClosedPath():
    def __init__(self, npts = 500, bias = 1.0, order = 4, line_width = 3,
                 center = [0, 0], amp = 1):
        self.ts = []
        self.rs = []
        self.xs = []
        self.ys = []
        self.weights = []
        self.phases = []
        self.order = order
        self.npts = npts
        self.bias = bias
        self.line_width = line_width
        self.center = center
        self.amp = amp
        self.gen_weights()
        self.weights = self.normalize_weights(self.weights)
        self.gen_phases()
        self.gen_path()

    def normalize_weights(self, weights):
        sum = 0
        for x in weights:
            sum += abs(x)
        weights = weights / sum
        return weights

    def path_radius(self, weights, phases, theta, bias):
        n = np.shape(weights)[0]
        fn = 0
        for i in range(n):
            fn += self.amp * weights[i]*np.cos(i * theta + phases[i])
        fn += bias
        return fn

    def dr_dtheta(self, weights, phases, theta, bias):
        n = np.shape(weights)[0]
        df = 0
        for i in range(n):
            df += -1.0 * self.amp * weights[i]*np.sin(i * theta + phases[i]) * i
        return df

    def get_theta_from_xy(self, pos):
        x_diff = pos[0] - self.center[0]
        y_diff = - pos[1] + self.center[1] # neg b/c \hat{y} points downward
        return np.arctan2(y_diff, x_diff)

    def tangent_angle(self, theta):
        r = self.path_radius(self.weights, self.phases, theta, self.bias)
        dr_dt = self.dr_dtheta(self.weights, self.phases, theta, self.bias)
        num = dr_dt * np.sin(theta) + r * np.cos(theta)
        den = dr_dt * np.cos(theta) - r * np.sin(theta)
        phi = np.arctan2(num, den) + np.pi
        return phi

    def gen_weights(self):
        for i in range(self.order):
            self.weights.append(np.random.rand() * 2 - 1)
        self.weights = np.array(self.weights)
        return self.weights

    def gen_phases(self):
        for i in range(self.order):
            self.phases.append(np.random.rand() * 2 * np.pi)
        self.phases = np.array(self.phases)
        return self.phases

    def get_x(self, t):
        x = self.path_radius(self.weights, self.phases, t, self.bias)*np.cos(t)
        x += self.center[0]
        return x

    def get_y(self, t):
        y = self.path_radius(self.weights, self.phases, t, self.bias)*np.sin(t)
        y += self.center[1]
        return y

    def gen_path(self):
        for t in np.linspace(0, 2* np.pi, self.npts):
            self.ts.append(t)
            self.rs.append(self.path_radius(self.weights, self.phases, t, self.bias))
            self.xs.append(self.get_x(t))
            self.ys.append(self.get_y(t))
        self.ts = np.array(self.ts)
        self.rs = np.array(self.rs)
        self.xs = np.array(self.xs)
        self.ys = np.array(self.ys)
        return None

    def draw_path(self, screen, color):
        pts = [self.xs, self.ys]
        pts = np.array(pts)
        self.outline = pts.T.astype(int)
        self.drawn_curve = pygame.draw.lines(screen, color, False, tuple(self.outline), self.line_width)
