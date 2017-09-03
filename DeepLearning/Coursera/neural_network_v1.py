import numpy as np

m = 4
print("Size of DataSet: \n",m)
n = 2
print("Number of Features: \n", n)

x = np.array([[1, 1],
			  [1, 0],
			  [0, 1],
			  [0, 0]])

print("Features: \n", x)

y = np.array([[1],
			  [1],
			  [1],
			  [0]])
print("\nLabels: \n", y)

def sigmoid(Z):
	"""
		- Mean closer to 0.5
		- Used in binary classification
		- Output layer
	"""
	return 1 / (1 + np.exp(-Z))

def d_sigmoid(Z):
	"""
		Derivative of sigmoid activation function
	"""
	return (sigmoid(Z)) * (1 - sigmoid(Z))

def tanh(Z):
	"""
		- Centers data
		- Mean closer to zero
	"""
	return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

def d_tanh(Z):
	"""
		Derivative of tanh activation function
	"""
	return 1 - (tanh(Z))**2

def reLU(Z):
	"""
		- Most popular activation function
	"""
	return np.maximum(0, Z)

def neural_network(x, y, m, n, L = 3, a = 0.004):
	"""
		x: Features
		y: Labels
		m: Size of Dataset
		n: Number of Features
		L: Layers (input -> hidden -> output)
		b: bias
		a: learning rate
	"""
	print("\nShape of Features: ", x.shape)
	print("\nShape of Labels:   ", y.shape)

	# Hidden Units
	hidden_units = 4

	# Weights for Layer 1
	w1 = np.random.randn(hidden_units, n)
	print("\nw1 shape = ", w1.shape)

	# Weights for Output Layer
	w2 = np.random.randn(1, hidden_units)
	print("\nw2 shape = ", w2.shape)

	# Bias for Layer 1
	b1 = np.zeros((hidden_units, 1)) * 0.01
	print("\nb1 shape = ", b1.shape)

	# Bias for Output Layer
	b2 = 0

	# Forward Propagation

	# Hidden Units not activated
	Z1 = w1 * x + b1
	print("\nHidden Units unactivated: \n", Z1)

	# Hidden Units Activated
	A1 = tanh(Z1)
	print("\nHidden Units activated: \n", A1)

	# Output Hidden Units Not Activated
	Z2 = w2.T * A1 + b2
	print("\nHidden Units Layer 2 unactivated: \n", Z2)

	A2 = sigmoid(Z2)
	print("\nHidden Units Layer 2 Activated: \n", A2)

	# Begin Backpropagation

	dZ2 = A2 - y
	print("\nOutput Layer Difference: \n", dZ2)

	dw2 = (1 / m) * dZ2.dot(A1.T)
	print("\nDerivative of output weights \n", dw2)

	# Cost of output layer
	db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
	print("\nOutput Layer Cost: \n", dZ2)

	# Cost of hidden Layer
	dZ1 = (w2.dot(dZ2)) * (d_tanh(Z1))
	print("\nCost of hidden layer: \n", dZ1)

	# Weight Derivative of Hidden Layer
	dw1 = (1 / m) * dZ1.dot(x.T)
	print("\nWeight derivatives \n", dw1)

	# Bias derivative
	db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
	print("\nBias Derivative \n", db1)


neural_network(x, y, m, n)






























