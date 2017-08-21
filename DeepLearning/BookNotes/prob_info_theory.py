print("Chapter 3 Probability and Information Theory")
print("Resources: Jaynes 2003")
print("Probability Theory: Represents uncertain statements.")
print(" - Reason in the presence of uncertainty.")
print(" - Tells how AI systems should reason.")
print(" - Analyze behavior")

print("\n\nThree possible sources of uncertainty:")
print("	1.) Creation of theoretical scenarios that postulate random dynamics")
print(" 2.) Deterministic systems that seem stochastic (game show example)")
print(" 3.) Dropout of features")

print("\n\nDegree of Belief or Bayesian Probability")
print(" 1 - indicating with absolute certainty that patient has flu")
print(" 0 - Indicating with absolute certainty that patient does not have the flu")

print("\n\nFrequentist Probability")
print(" - Probability of drawing a certain hand in poker")


print("\n\nVariables:\n")
print("Discrete Variables: Finite number of states")
print("Continuous Variables: Infinite number of values or states\n\n")

print("Probability Distribution: How likely a random variable is to take on a state\n")
print("depending on whether its discrete or continuous.\n\n\n")

print("Probability Mass Function")
print("	- Probability of descrete variable to take on that state")
print(" 	Probability of x taking on 'x' ")
print(" 	* P(x)")
print(" 	* P(x = x)")
print(" 	* x ~ P(x)\n")

print("Join Probability Distribution: Probability Mass function of MULTIPLE variables")
print("		* P(x = x, y = y)")
print("		* P(x, y)\n")

print("Properties of Probability Mass Functions:")
print(" * Domain of P must be the set of all possible states of x")
print(" * 0: Impossible State >= P(X) <= 1: Gauranteed State")
print(" * Normalized Property: For all values x, P(x) = 1\n\n")

print("Uniform Distribution: Make k number of states for variable x equally likely =>")
print("P(x = x) = 1 / K for all i =>")
print("for all P(x - xi) = for all 1/ k = k / k = 1 \n\n\n")

print("Continuous Variables and Probability Density Functions")
print("Probability Density Function: Used when working with continuous variables.")
print("  - Domain of p must be set to all possible states of x")
print("  - For all values x, there exists a value x where p(x) >= 0. x can be greater than 1")
print("  - f p(x)dx = 1  \n\n\n")

print("Marginal Probability: Probability Distribution of a subset")
print("For all values x, there exists an element x, where the probability of x is equal to")
print("the sum of y and the probability of x and y.  \n\n\n")

print("Conditional Probability: The probability of some event given that some other event happened.")
print("	- Given: y = y, x = x")
print(" - P(y = y, x = x) = P(y = y, x = x) / P(x = x)")
print(" - P(x = x) > 0 \n")

print("Intervetion Query: Computing the consequences of an action \n\n\n\n")

print("The Chain Rule of Conditional Probabilities:")
print(" - P(x=x|y=y) == P(x=x,y=y)/P(x=x)")

print("Independence and Conditional Independence")
print("Independent Variables")
print(" - Two random variables x and y are independent if their probability distribution can be")
print("   expressed as a product of two factors, one involving only x and one involving y")
print(" - For all values x, there exists an element x and an element y where the probability of ")
print("   x and y are equal to P(x=x)p(y=y) \n\n")

print("Conditionally Independent:")
print(" - For all values x, there exists an element x, y and z, where the probability of x and the conditional probability")
print("   of y and z, P(x=x,y=y|z=z), is equal to the conditional probability of P(x|z) times the conditional probability")
print("   of P(y|z) \n\n\n")

print("Expectation, Variance, and Covariance")
print(" * Expectation or Expected Value for Discrete Variables:")
print("     - The sum of the probability of x and some function(x) for all random values of x\n\n")

print(" * Expectation or Expected Value for Continuous Variables:")
print("     - Indefinite Integral(reverse derivative) of the probability of x times some function x and the derivative of x.\n\n")


print("Variance: gives a measure of how much the values of a function of a random variable x vary as")
print(" we sample different values of x from its probability distribution:")
print(" Var(f(x)) = The expected value of the some function x minus the expected value of some function x squared")
print("             E[ (f(x) - E[f(x)])^2 ]  \n\n\n\n\n")


print("Covariance: Gives some sense as to how much two variables are linearly related")
print("Cov(f(x),g(y)) = The expected value of some function x minus the expected value of some function x times another function y")
print("				    minus the expected value of another function y.")
print("               = E [(fx) - E[f(x)])(g(y) - E[g(y)])]   \n")

print("A high covariance means the values change very much and are far from their means at the same time.")
print("If covariance is positive, both variables take on high values.")
print("If covariance is negative, one variable takes high values while other takes low values.\n\n")

print("Correlation: normalize the contribution of each ariable in order to measure only how much the variables are related, rather")
print(" than also being affected by the scale of the separate variables.\n\n\n")

print("Covariance and Dependence are related because: ")
print(" - Two variables that are independent have zero covariance, and two variables that are non-zero covariance are dependent.\n\n")

print("Covarinace and Dependence are not related because: ")
print(" - For two variables to have zero covariance, there must be no linear dependence between them.")
print(" - It is possible for two variables to be dependent but have zero covariance.\n\n\n")

print("Covariance Matrix: of a random vector x for all real n elements is an n x n matrix such that,")
print(" Cov(x)i,j = Cov(xi, xj) \n\n\n")

print("The diagonal elements of the covariance give the variance:")
print(" Cov(xi,xi) = Var(xi) \n\n\n")

print("Bernoulli Distribution: ")
print(" - Distribution over a single binary random variable [0, 1]")
print(" - Gives probability of the random variable being equal to 1.\n\n")

print("Multinoulli Distribution")
print(" - Distribution over a single discrete variable with k different states, where k is finite.")
























































