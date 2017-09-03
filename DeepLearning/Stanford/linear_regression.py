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
		b_gradient += - (2 / N) * (y - ((m_current * x) + b_current))
		m_gradient += - (2 / N) * x * (y - ((m_current * x) + b_current))
	new_b = b_current - (l_rate * b_gradient)
	new_m = m_current - (l_rate * m_gradient)
	return [new_b, new_m]

def gradient_descent(points, starting_b, starting_m, l_rate, n_iterations):
	b = starting_b
	m = starting_m
	for i in range(n_iterations):
		b, m = step_gradient(b, m, array(points), l_rate)
	return [b, m]

def gradient_descent_epsilon(points, starting_b, starting_m, n_iterations, EPSILON=0.0001):
	# for i in range(1, len(points)):
	# 	theta_p = theta
	# 	theta_p[i] = theta_p[i] + EPSILON
	# 	theta_m = theta
	# 	theta_m[i] = theta_m[i] - EPSILON
	# 	grad_approx[i] = (J(theta_plus) - J(theta_minus)) / (2 * EPSILON)	

def run():
    points = genfromtxt("data.csv", delimiter=",")
    print("points:")
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, squared_error(initial_b, initial_m, points)))
    print("Running...")
    [b, m] = gradient_descent(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, squared_error(b, m, points)))

if __name__ == "__main__":
	run()