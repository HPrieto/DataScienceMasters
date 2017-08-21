import math
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
	ex = np.exp(-x)
	return ex / (1 + ex) ** 2

A = np.random.random_integers(10, size=(5, 5))

P = np.linalg.pinv(A)

I = np.round(np.matmul(P, A))

t = np.arange(0,1, 0.01)

PI = np.pi

sin = np.sin(2 * PI * 4 * t)

cos = np.cos(2 * PI * 4 * t)

tan = np.tan(2 * PI * 2 * t)

x = np.array([1000, 1500, 2000, 2500, 3000])

y = np.array([100000, 120000, 180000, 210000, 250000])

x_t = x.transpose()

rise = np.matmul(y,x_t) / 100000

run = np.matmul(x, x_t) / 100000

m = rise / run

print("rise:  ", rise)
print("run:   ", run)
print("slope: ", m)

print("A * A-1 = ", I)

print("\n\n\nNormal Equation Practice: ")

X = np.array([[1, 2104, 5, 1, 45],
			  [1, 1416, 3, 2, 40],
			  [1, 1534, 3, 2, 30],
			  [1,  852, 2, 1, 36]])

y = np.array([460, 232, 315, 178])

X_t = X.transpose()

n_theta_inv = np.linalg.pinv(X_t.__matmul__(X))

n_theta_xy = np.matmul(X_t,y)

n_theta = np.matmul(n_theta_inv, n_theta_xy)

print("Features: ")
print(X)
print("\n\nFeatures Transposed:")
print(X_t)
print("\n\nOutputs:")
print(y)
print("-(X^T * X) : ")
print(n_theta_inv)
print("\n\n X^T * y :")
print(n_theta_xy)
print("\n\nNormal Equation Parameters: ")
print(n_theta)












