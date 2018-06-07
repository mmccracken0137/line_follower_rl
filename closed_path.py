import numpy as np
import matplotlib.pyplot as plt

class ClosedPath():
    def __init__(self, npts = 200, bias = 1.0, order = 4):
        self.ts = []
        self.rs = []
        self.xs = []
        self.ys = []
        self.weights = []
        self.phases = []
        self.order = order
        self.npts = npts
        self.bias = bias
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

    def path_fn(self, weights, phases, theta):
        n = np.shape(weights)[0]
        fn = 3.0
        for i in range(n):
            fn += weights[i]*np.cos(i * theta + phases[i])
        fn = fn * fn
        return fn

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

    def gen_path(self):
        for t in np.linspace(0, 2* np.pi, 100):
            self.ts.append(t)
            self.rs.append(self.path_fn(self.weights, self.phases, t))
            self.xs.append(self.rs[-1]*np.cos(self.ts[-1]))
            self.ys.append(self.rs[-1]*np.sin(self.ts[-1]))
        self.ts = np.array(self.ts)
        self.rs = np.array(self.rs)
        self.xs = np.array(self.xs)
        self.ys = np.array(self.ys)
        return None
