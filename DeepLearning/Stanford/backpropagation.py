import numpy as np

# Defined binary input data
X = np.array([[0, 0, 1],
			  [0, 1, 1],
			  [1, 0, 1],
			  [1, 1, 1]])

y = np.array([[0],
			  [1],
			  [1],
			  [0]])

print("Training Set:")
print(X)
print("\nLabels:")
print(y)

# Weights
np.random.seed(1)
weights0 = 2 * np.random.random((3,4)) - 1
weights1 = 2 * np.random.random((4,1)) - 1

print("\n\nNeural Net Layer 1 -> Layer 2 Weights: ")
print(weights0)
print("\n\n\n\nNeural Net Layer 2 -> Output Layer Weights: ")
print(weights1)


# Sigmoid - Activation Function
def nonlin(x, deriv=False):
	if (deriv == True):
		return x * (1 - x)
	return 1 / (1 + np.exp(-x))

# Begin training
for j in range(60000):
	# Feed Forward through layers: 1, 2, 3
	a0 = X
	a1 = nonlin(np.dot(a0, weights0))
	a2 = nonlin(np.dot(a1, weights1))

	if (j % 10000) == 0:
		print("\n\n\n\nTraining Iteration #", j)
		print("Begin forward propagation:")
		print("a1 = ")
		print(a1)
		print("\na2 = ")
		print(a2)

	# How much did we miss the target value?
	a2_error = y - a2

	# In what direction is the target value?
	# were we really sure? if so, don't change too much.
	a2_delta = a2_error * nonlin(a2, deriv=True)

	# How much did each k1 value contribute to the k2 error (according to the weights)
	a1_error = a2_delta.dot(weights1.T)

	# In what direction is the target k1?
	# Were we really sure? if so, don't change too much.
	a1_delta = a1_error * nonlin(a1, deriv=True)

	if (j % 10000) == 0:
		print("\nBegin Backpropagation:")
		print("Output delta: ")
		print(a2_delta)

		print("\nLayer 2 delta: ")
		print(a1_delta)

	weights1 += a1.T.dot(a2_delta)
	weights0 += a0.T.dot(a1_delta)

	if (j % 10000) == 0:
		print("\nLayer 1 -> Layer 2 Weights after Backpropagation: ")
		print(weights0)
		print("\nLayer 2 -> Output Layer Weights after Backpropagation: ")
		print(weights0)
