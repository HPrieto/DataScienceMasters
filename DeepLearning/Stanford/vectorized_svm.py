import numpy as np

def L_i_vectorized(x, y, W):
	"""
		Vectorized Loss Function:
		Li = ∑i≠yi max(0, sj - syi + 1)
		x : image
		y : label
		W : weights
	"""
	scores = W.dot(x)
	margins = np.maximum(0, scores - scores[y] + 1)
	margins[y] = 0
	loss_i = np.sum(margins)
	return loss_i

def vanilla_gradient_descent(data, weights, loss_fun, step_size):
	while True:
		data_batch = sample_training_data(data, 256) # sample 256 examples
		weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
		weights += - step_size * weights_grad # perform parameter update