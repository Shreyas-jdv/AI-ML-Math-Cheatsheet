import numpy as np


# Vectors
def dot_product(a, b):
    return np.dot(a, b)


def cross_product(a, b):
    return np.cross(a, b)


def norm(a):
    return np.linalg.norm(a)


# Matrices
def matrix_addition(A, B):
    return np.add(A, B)


def matrix_multiplication(A, B):
    return np.matmul(A, B)


def transpose(A):
    return np.transpose(A)


def inverse(A):
    return np.linalg.inv(A)


def determinant(A):
    return np.linalg.det(A)


# Eigenvalues and Eigenvectors
def eigen(A):
    return np.linalg.eig(A)



import sympy as sp


# Derivatives
def gradient(f, vars):
    return [sp.diff(f, var) for var in vars]


def chain_rule(dy_du, du_dx):
    return dy_du * du_dx


def partial_derivative(f, var):
    return sp.diff(f, var)


# Integrals
def definite_integral(f, var, a, b):
    return sp.integrate(f, (var, a, b))


def indefinite_integral(f, var):
    return sp.integrate(f, var)



from scipy.stats import norm, binom, poisson


# Probability
def conditional_probability(P_A_and_B, P_B):
    return P_A_and_B / P_B


def bayes_theorem(P_B_given_A, P_A, P_B):
    return (P_B_given_A * P_A) / P_B


def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))


# Distributions
def normal_distribution(x, mu, sigma):
    return norm.pdf(x, mu, sigma)


def binomial_distribution(n, k, p):
    return binom.pmf(k, n, p)


def poisson_distribution(k, lam):
    return poisson.pmf(k, lam)


# Hypothesis Testing
def z_score(X, mu, sigma):
    return (X - mu) / sigma


def t_score(X, mu, S, n):
    return (X - mu) / (S / np.sqrt(n))


# p-value calculation (requires context, typically done using scipy or statsmodels)


# Gradient Descent
def gradient_descent(theta, alpha, gradient):
    return theta - alpha * gradient


# Lagrange Multipliers
def lagrange_multipliers(grad_f, grad_g):
    return grad_f - grad_g


# Stochastic Gradient Descent
def stochastic_gradient_descent(theta, alpha, gradient):
    return theta - alpha * gradient  # Similar to gradient descent but applied to each sample



# Perceptron
def perceptron_activation(w, x, b):
    return np.sign(np.dot(w, x) + b)


# Support Vector Machines
def margin(w):
    return 2 / np.linalg.norm(w)


# Logistic Regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_loss(y, y_hat):
    return - (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


# Example of dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product_result = dot_product(a, b)

# Example of matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
matrix_multiplication_result = matrix_multiplication(A, B)

# Example of calculating the gradient
x, y = sp.symbols('x y')
f = x ** 2 + y ** 2
gradient_result = gradient(f, [x, y])

# Print results
print(f"Dot product: {dot_product_result}")
print(f"Matrix multiplication result:\n{matrix_multiplication_result}")
print(f"Gradient: {gradient_result}")



def print_formulas():
    # Linear Algebra
    print("1. Linear Algebra")
    print("Vectors:")
    print("Dot Product: a · b = Σ (a_i * b_i)")
    print("Cross Product (for 3D vectors): a × b = (a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1)")
    print("Norm (Magnitude): ||a|| = sqrt(Σ (a_i^2))")

    print("\nMatrices:")
    print("Matrix Addition: A + B")
    print("Matrix Multiplication: C = AB, where C_ij = Σ (A_ik * B_kj)")
    print("Transpose: A^T")
    print("Inverse: A^(-1), such that AA^(-1) = I")
    print("Determinant (for a 2x2 matrix): det(A) = ad - bc")
    print("Eigenvalues and Eigenvectors:")
    print("Eigenvalue Equation: Av = λv, where A is a matrix, v is an eigenvector, and λ is an eigenvalue.")

    # Calculus
    print("\n2. Calculus")
    print("Derivatives:")
    print("Gradient: ∇f(x) = [∂f/∂x1, ∂f/∂x2, ..., ∂f/∂xn]")
    print("Chain Rule: dy/dx = (dy/du) * (du/dx)")
    print("Partial Derivatives: ∂f/∂xi")

    print("\nIntegrals:")
    print("Definite Integral: ∫_a^b f(x) dx")
    print("Indefinite Integral: ∫f(x) dx = F(x) + C")

    # Probability and Statistics
    print("\n3. Probability and Statistics")
    print("Probability:")
    print("Conditional Probability: P(A | B) = P(A ∩ B) / P(B)")
    print("Bayes’ Theorem: P(A | B) = [P(B | A) * P(A)] / P(B)")
    print("Expected Value: E[X] = Σ (x_i * P(x_i))")

    print("\nDistributions:")
    print("Normal Distribution: f(x | μ, σ^2) = (1 / sqrt(2πσ^2)) * exp(-(x - μ)^2 / (2σ^2))")
    print("Binomial Distribution: P(X = k) = (n choose k) * p^k * (1 - p)^(n - k)")
    print("Poisson Distribution: P(X = k) = (λ^k * e^(-λ)) / k!")

    print("\nHypothesis Testing:")
    print("Z-Score: Z = (X - μ) / σ")
    print("T-Score: T = (X - μ) / (S / sqrt(n))")
    print(
        "p-value: The probability of obtaining results at least as extreme as the observed results, under the assumption that the null hypothesis is true.")

    print("\nRegression:")
    print("Simple Linear Regression: y = mx + c")
    print("Multiple Linear Regression: y = β0 + β1x1 + ... + βnxn")

    print("\nCorrelation and Covariance:")
    print("Covariance: cov(X, Y) = E[(X - E[X])(Y - E[Y])]")
    print("Pearson Correlation Coefficient: ρX,Y = cov(X, Y) / (σX * σY)")

    # Optimization
    print("\n4. Optimization")
    print("Gradient Descent Update Rule: θ = θ - α∇θJ(θ)")
    print("Lagrange Multipliers: ∇f(x) = λ∇g(x)")
    print("Stochastic Gradient Descent: Similar to Gradient Descent but updates parameters for each training example.")

    # Linear Models
    print("\n5. Linear Models")
    print("Perceptron Activation Function: sgn(w · x + b)")
    print("Support Vector Machines Margin: Margin = 2 / ||w||")
    print("Logistic Regression Sigmoid Function: σ(z) = 1 / (1 + e^(-z))")
    print("Logistic Regression Loss Function: L(y, ŷ) = -[y log(ŷ) + (1 - y) log(1 - ŷ)]")

# Call function to print all the formulas
print_formulas()