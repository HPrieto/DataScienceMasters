m = 10
n = 20
p = 30

def createMatrix(r,c):
	return [[x +1 for x in range(r)] for y in range(c)]

def printMatrix(m):
	rows = len(m)
	cols = len(m[0])
	for i in range(rows):
		print(m[i])

A = createMatrix(m, n)
B = createMatrix(n, p)
C = createMatrix(m, p)

print("A Matrix:")
printMatrix(A)

print("B Matrix:")
printMatrix(B)

print("C Matrix:")
printMatrix(C)

print("Orthogonal Matrix: A^-1 == A^T")

print("Eigendecomposition")
print("Eigen Vector * Eigen Scalar == A * Eigen Vector\n\n")

print("Singular Value Decomposition")
print("A = UDV")

print("A: m * n")

print("U: m * m")
print("D: m * n")
print("V: n * n\n\n")

print("U(Orthogonal): Columns known as Left-singular values")
print("D(Diagonal)  : Elements along diagonal A known as singular values")
print("V(Orthogonal): Columns known as Right-singular values")

print("\n\n\n\n")
print("The Moore-Penrose Pseudoinverse")
print("A+ = Pseudoinverse of Matrix A")
print("A+ = V * D+ * U^T")


print("\n\n\n\n\n")
print("Trace Operator:")
print("Tr(A): Sum of all diagonal values of matrix A")
print("Tr(A) == Tr(A^T)")

print("\n\n\n\n\n\n")
print("The Determinate")
print("Product of all the eigenvalues")
print("Measure of how much multiplication by the matrix expands or contracts space")
print("If the determinant == 0: Then volume is completely lost")

print("Measure distance between x and c* using norm:")
print("(x i g(x))^T (x - g(c)) =>")
print("(x^T * x) - (x^Tg(c)) - (g(c)^Tg(c)) <= using distribution")



U = createMatrix(m, m)
D = createMatrix(m, n)
V = createMatrix(n, n)

print("\n\n\nA Matrix:")
printMatrix(A)

print("U Matrix:")
printMatrix(U)

print("D Matrix:")
printMatrix(D)

print("V Matrix:")
printMatrix(V)