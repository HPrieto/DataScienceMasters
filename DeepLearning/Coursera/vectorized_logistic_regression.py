import numpy as np

def sigmoid(Z):
  return 1 / (1 + np.exp(-Z))

def logistic_regression(n = 10, m = 10, d = 2, a = 0.004):
	"""
		n : features
		m : test dataset
		d : classification dimension
	"""

	# Vectorized Features
	X = np.random.randint(d, size=(n,m))
	print("Features: \n", X)

	# Vectorized Labels
	y = np.random.randint(d, size=(n,1))
	print("\nLabels: \n", y)

	# Vectorized Weights
	w = np.random.rand(n)
	print("\nWeights: \n", w)

	# Bias (random real number)
	b = 1

	# Iterate 10000 times
	for i in range(10000):
		
		# Vectorized Parameters
		Z = np.dot(w.T, X) + b
		print("\nParameters: \n", Z)

		# Parameters Activated
		A = sigmoid(Z)
		print("\nActivated Parameters: \n", A)

		# Derivative of Parameters
		dZ = A - y
		print("\nDerivative of Parameters: \n", dZ)

		# Vectorized Forward Propagation
		dw = (1 / m) * np.dot(X, dZ.T)
		print("\nForward Propagation Result: \n", dw)

		# Vectorized Back Propagation
		db = (1 / m) * np.sum(dw)
		print("\nDerivative of the bias: \n", db)

		# Update weights
		w = w - np.dot(a, dw)
		print("\nUpdated Weights: \n", w)

		# Update bias
		b = b - (a * db)
		print("\nUpdated bias: \n", b)

	print("\nFeatures: \n",X)
	print("\nLabels: \n", y)


logistic_regression()