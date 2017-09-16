import numpy as np

H = 5	# dimensionality of hidden state
T = 50  # number of time steps
Whh = np.random.randn(H,H)

# forward pass of an RNN(ignoring inputs x)
hs = {}
ss = {}
hs[-1] = np.random.randn(H)
for t in xrange(T):
	ss[t] = np.dot(Whh, hs[t - 1])
	hs[t] = np.maximum(0, ss[t])

# backward pass of the RNN
dhs = {}
dss = {}
dhs[T-1] = np.random.randn(H) # start off the chain with random gradient
for t in reversed(xrange(T)):
	dss[t] = (hs[t] > 0) * dhs[t]   # backprop through nonlinearity
	dhs[t - 1] = np.dot(Whh.T, dss[t]) # backprop into previous hidden state 