import numpy as np

def loss_function(y_hat, y):
	return 0.5 * (y_hat - y) ** 2

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def cost_function(y_hats, ys, m):
	cost_sum = 0
	for i in xrange(m):
		cost_sum = cost_sum + loss_function(y_hats[i],ys[i])
	return (1 / m) * cost_sum

"""

Vectorized implementation practice

"""

a = np.random.rand(1000000)
b = np.random.rand(1000000)

# Vectorized w†x + b
c = np.dot(a, b)

# Exponential
# v = [v1,   ..., vn] -> u = [e^v1, ..., e^vn]

v = np.random.randint(10, size=(10))
u = np.exp(v)

print("v = " + str(v))
print("u = " + str(u) + "\n\n\n\n")

def vectorized_z(w, x, b):
	return np.dot(w.T, x) + b

def vectorized_sigmoid(z):
	return np.sigmoid(z)

"""
	Broadcasting example
"""

A = np.array([[56.0, 0.0,   4.4,  68.0],
			  [1.2,  104.0, 52.0, 8.0],
			  [1.8,  135.0, 99.0, 0.9]])
print("Matrix A: \n\n")
print(A)

cal = A.sum(axis=0)

print("\nCal: \n")
print(cal)

percentage = 100 * A / cal
print("\nPercentage:")
print(percentage)
"""
Notes:

Given x, want y^ = P(y = 1 | x)

x : n * 1 , y : [0, 1]

X = n * m matrix

Parameters : w : n * 1 , b : real number

z = w† * x(¡) + b

¡ = data index

output y^ = S(z)

a(¡) = S(z(¡))

S(z) = 1 / (1 + e ** -z)

if z == large : S(z) ~ 1
if z == small : S(z) ~ 0

Loss Function : L(y^, y) = 0.5 * (y^ - y) ** 2
	* Measures how well we are doing on a single training example

L(y^, y) = -(y log y^ + (1 - y) log (1 - y^))

if y == 1:
	L(y^, y) : want y as close to 1 as possible
else if y == 0:
	L(y^, y) : want y as close to 0 as possible


Cost Function : J(w, b) = (1 / m) ∑ L( y^(¡), y(¡) )
	* Measures how well out parameters (w, b) are doing on out entire training set

w : weight : initialize to 0 or random
b : bias   : initialize to 0 or random

Gradient Descent:
∂ : derivative term (How much the function slopes)
å : learning rate [-0.001, 0.001]
parameters (w) : Repeat {
	w := w - å * ( å * ( (∂ * J(w)) / (∂ * w) )
	 * shorthand == w := w - å∂w
}

parameters (w, b) Repeat {
	w := w - å * ( å * ( (∂ * J(w, b)) / (∂ * w) )

	b := b - å * ( å * ( (∂ * J(w, b)) / (∂ * w) )
}

Logistic Regression Example:

x  = R^n
y  = R
y  = {}
z  = W†x + b
y^ = a = S(z)
£  = -(ylog(a) + (1 - y)log(1 - a))
w  = R^n : random values or 0
b  = some real number

Compute Example:
z  = w1x1 + w2x2 + b
y^ = S(z)
£  = £(y^, y)

Logistic Regression:
	m = size of training set
	n = size of features in x
	J=0; dw1=0; dw2=0; dwN=0; db=0;
	w = np.array.rand(n)
	b = math.rand()
	for i in range(m):
		z[i] = w.T * x[i] + b
		a[i] = (z[i])
		J = -[y[i]*log(a[i]) + (1 - y[i]) * log(1 - a[i])]
		dz[i] = a[i] - y[i]
		dw1 += x1[i] * dz[i]
		dw2 += x2[i] * dz[i]
			... for all features
		dwN += xN[i] * dz[i]
		db  += dz[i]
	J \= m
	dw1 \= m
	dw2 \= m
	dwN \= m
	db \= m

	Repeat {
		w1 = w1 - å * dw1
		w2 = w2 - å * dw2
			... for all features in x
		wn = wn - å * dwN
		b  = b - å * db
	}

"""








































