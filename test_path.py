import numpy as np
import matplotlib.pyplot as plt

from closed_path import *

path = ClosedPath(100, 1.0, 5)
plt.plot(path.xs, path.ys)
plt.show()
