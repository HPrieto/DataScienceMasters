import numpy as np
import matplotlib.pyplot as plt

n = 2

m = 200

X = np.ones(m)

def linear_features(X, m):
	for i in range(m):
		X[i] = i * 2

linear_features(X, m)

y = np.random.randint(10, size=(m, 1))

print('Features: \n', X)

print('Labels: \n', y)

plt.plot(X, 'b', y, 'g')
plt.show()