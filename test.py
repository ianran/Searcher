# simple test

import numpy as np

file = np.load(feed.npz)

x = file['x']
y = file['y']

print(x[0])
print(y[0])
