from numpy import *

# y = mx + b
# m is slope, b is y-intercept
def squared_error(b, m, points):
	error = 0
	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]
		error += (y - (m * x + b)) ** 2
	return error / float(len(points))

def step_gradient(b_current, m_current, points, l_rate):
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))
	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]
		b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
		m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))
	new_b = b_current - (l_rate * b_gradient)
	new_m = m_current - (l_rate * m_gradient)
	return [new_b, new_m]

def gradient_descent(points, starting_b, starting_m, l_rate, n_iterations):
	b = starting_b
	m = starting_m
	for i in range(n_iterations):
		b, m = step_gradient(b, m, array(points), l_rate)
	return [b, m]