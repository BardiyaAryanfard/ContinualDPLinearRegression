import numpy as np


class Matrix:
    def __init__(self, n_rows, n_columns, values=None):
        if values is None:
            self.data = np.zeros((n_rows, n_columns))
        else:
            self.data = np.array(values)
        self.n_rows, self.n_columns = self.data.shape

    def add(self, other):
        if self.data.shape != other.data.shape:
            raise ValueError("Matrices must have the same dimensions to add.")
        return Matrix(self.n_rows, self.n_columns, self.data + other.data)

    def subtract(self, other):
        if self.data.shape != other.data.shape:
            raise ValueError("Matrices must have the same dimensions to subtract.")
        return Matrix(self.n_rows, self.n_columns, self.data - other.data)

    def multiply(self, other):
        if self.n_columns != other.n_rows:
            raise ValueError("Number of columns of the first matrix must equal the number of rows of the second.")
        return Matrix(self.n_rows, other.n_columns, np.dot(self.data, other.data))

    def multiply_by_num(self, num):
        return Matrix(self.n_rows, self.n_columns, self.data * num)

    def transpose(self):
        return Matrix(self.n_columns, self.n_rows, self.data.T)

    def first_rows(self, num):
        if num > self.n_rows:
            raise ValueError("Number of rows requested exceeds matrix dimensions.")
        return Matrix(num, self.n_columns, self.data[:num, :])

    def inverse(self):
        if self.n_rows != self.n_columns:
            raise ValueError("Matrix must be square to calculate its inverse.")
        return Matrix(self.n_rows, self.n_columns, np.linalg.inv(self.data))

    def dimensions(self):
        return self.n_rows, self.n_columns

    def min_eigenvalue(self):
        if self.n_rows != self.n_columns:
            raise ValueError("Matrix must be square to calculate eigenvalues.")
        eigenvalues = np.linalg.eigvals(self.data)
        return np.min(eigenvalues)

    def to_array(self):
        return self.data.tolist()


def generate_symmetric_gaussian_matrix(n, variance):
    # Calculate standard deviation from variance
    sigma = np.sqrt(variance)
    # Create an n x n matrix with i.i.d. Gaussian random variables
    upper_triangular = np.random.normal(0, sigma, size=(n, n))
    # Make the matrix symmetric
    symmetric_matrix = np.triu(upper_triangular) + np.triu(upper_triangular, 1).T
    # Return as a Matrix object
    return Matrix(n, n, symmetric_matrix)


def generate_gaussian_matrix(n_rows, n_columns, variance):
    if n_rows == n_columns:
        return generate_symmetric_gaussian_matrix(n_rows, variance)
    # Calculate standard deviation from variance
    sigma = np.sqrt(variance)
    # Create an n_rows x n_columns matrix with i.i.d. Gaussian random variables
    gaussian_matrix = np.random.normal(0, sigma, size=(n_rows, n_columns))
    # Return as a Matrix object
    return Matrix(n_rows, n_columns, gaussian_matrix)


def identity(size):
    return Matrix(size, size, np.identity(size))
