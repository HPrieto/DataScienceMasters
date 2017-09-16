import numpy as np

def init_two_layer_model(input_size, hidden_size, output_size):
	# initialize model
	model = {}
	model['W1'] = 0.0001 * np.random.randn(input_size, hidden_size)
	model['b1'] = np.zeros(hidden_size)
	model['W2'] = 0.0001 * np.random.randn(hidden_size, output_size)
	model['b2'] = np.zeros(output_size)
	return model

# input size, hidden size, number of classes
model = init_two_layer_model(32*32*32, 50, 10)
loss, grad = two_layer_net(X_train, model, y_train, 1e3)