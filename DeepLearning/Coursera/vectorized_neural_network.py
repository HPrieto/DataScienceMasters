import numpy as np

m = 4
print("Size of DataSet: \n",m)
n = 2
print("Number of Features: \n", n)

x = np.array([[1, 1],
			  [1, 0],
			  [0, 1],
			  [0, 0]])

# x = np.array([[1, 1, 1],
# 			  [1, 1, 0],
# 			  [1, 0, 1],
# 			  [1, 0, 0],
# 			  [0, 1, 1],
# 			  [0, 1, 0],
# 			  [0, 0, 1],
# 			  [0, 0, 0],])

print("Features: \n", x)

y = np.array([[1],
			  [1],
			  [1],
			  [0]])

# y = np.array([[1],
# 			  [1],
# 			  [1],
# 			  [0],
# 			  [0],
# 			  [0],
# 			  [0],
# 			  [0]])
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

def neural_network(X, Y, n, m, a=0.004):
	"""
		X : Entire training dataset (n * m)
		Y : Entire training labels (1 * m)
		n : Number of features
		m : Size of training set
		a : Learning Rate (0.004 standard)
	"""
	print("Features: ", n)
	print("Dataset:  ", m)

	# Number of Layers in Neural Network
	L = 3
	print("Neural Net Layers: ", L)

	# Number of Hidden nodes/units
	h = n ** 2
	print("Neural Net Hidden Units: ", h)

	# L1 Weights
	W1 = np.random.randn(h, n)
	print("Layer 1 Weights Shape: ", W1.shape)
	print("Layer 1 Weights: \n", W1)

	# L1 Biases
	b1 = np.zeros((h, n))
	print("Layer 1 Biases Shape: ", b1.shape)
	print("Layer 1 Biases: \n", b1)

	# L1 Z (Unactivated Values)
	Z1 = W1 * X + b1
	print("Layer 1 Unactivated Values Shape: ", Z1.shape)
	print("Layer 1 Z (unactivated values) \n", Z1)

	# L1 A (Activated Values)
	A1 = tanh(Z1)
	print("Layer 1 Activated Values Shape: ", A1.shape)
	print("Layer 1 A (activated values) \n", A1)

	# L2 Weights
	W2 = np.random.randn(h, h)
	print("Layer 2 Weights Shape: ", W2.shape)
	print("Layer 2 Weights: \n", W2)

	# L2 Biases
	b2 = np.zeros((h, n))
	print("Layer 2 Biases Shape: ", b2.shape)
	print("Layer 2 Biases: \n", b2)

	# L2 Z (Unactivated Values)
	Z2 = W2.dot(A1) + b2
	print("Layer 2 Unactivated Values Shape: ", Z2.shape)
	print("Layer 2 Z (unactivated values) \n", Z2)

	# L2 A (Activated Values)
	A2 = tanh(Z2)
	print("Layer 2 Activated Values Shape: ", A2.shape)
	print("Layer 2 A (activated values) \n", A2)

	# L3 Weights
	W3 = np.random.randn(m, h)
	print("Layer 3 Weights Shape: ", W3.shape)
	print("Layer 3 Weights: \n", W3)

	# L3 Biases
	b3 = np.zeros((m, n))
	print("Layer 3 Biases Shape: ", b3.shape)
	print("Layer 3 Biases: \n", b3)

	# L3 Z (Unactivated Values)
	Z3 = W3.dot(A2) + b3
	print("Layer 3 Unactivated Values Shape: ", Z3.shape)
	print("Layer 3 Z (unactivated values) \n", Z3)

	# L3 A (Activated Values)
	A3 = sigmoid(Z3)
	print("Layer 3 Activated Values Shape: ", A3.shape)
	print("Layer 3 A (activated values) \n", A3)

	# Cost
	dZ3 = A3 - Y
	print("Layer 3 Cost Shape: ", dZ3.shape)
	print("Layer 3 Cost Derivative: \n", dZ3)

	# Layer 3 Derivative of Weights
	dW3 = (1 / m) * dZ3 * A3
	print("Layer 3 Weights Shape: ", dW3.shape)
	print("Layer 3 Weights Derivative: \n", dW3)

	# Layer 3 Biases
	db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)
	print("Layer 3 Biases Shape: ", db3.shape)
	print("Layer 3 Biase Derivative: \n", db3)

	# Cost
	dZ2 = W3.dot(dZ3) * d_tanh(Z3)
	print("Layer 2 Cost Shape: ", dZ2.shape)
	print("Layer 2 Cost Derivative: \n", dZ2)

	# Layer 3 Derivative of Weights
	dW2 = (1 / m) * dZ2 * A2
	print("Layer 2 Weights Shape: ", dW2.shape)
	print("Layer 2 Weights Derivative: \n", dW2)

	# Layer 3 Biases
	db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
	print("Layer 2 Biases Shape: ", db2.shape)
	print("Layer 2 Biase Derivative: \n", db2)

	# Cost
	dZ1 = W2.dot(dZ2) * d_tanh(Z2)
	print("Layer 1 Cost Shape: ", dZ1.shape)
	print("Layer 1 Cost Derivative: \n", dZ1)

	# Layer 3 Derivative of Weights
	dW1 = (1 / m) * dZ1 * A1
	print("Layer 1 Weights Shape: ", dW1.shape)
	print("Layer 1 Weights Derivative: \n", dW1)

	# Layer 3 Biases
	db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
	print("Layer 1 Biases Shape: ", db1.shape)
	print("Layer 1 Biase Derivative: \n", db1)

neural_network(x, y, n, m)






























