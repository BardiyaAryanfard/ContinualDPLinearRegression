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

    def spectral_norm(self):
        if self.n_rows != self.n_columns:
            raise ValueError("Matrix must be square to calculate spectral norm.")
        singular_values = np.linalg.svd(self.data, compute_uv=False)
        return np.max(singular_values)

    def to_array(self):
        return self.data.tolist()


def generate_symmetric_gaussian_matrix(n, variance):
    sigma = np.sqrt(variance)
    upper_triangular = np.random.normal(0, sigma, size=(n, n))
    symmetric_matrix = np.triu(upper_triangular) + np.triu(upper_triangular, 1).T
    return Matrix(n, n, symmetric_matrix)


def generate_gaussian_matrix(n_rows, n_columns, variance):
    if n_rows == n_columns:
        return generate_symmetric_gaussian_matrix(n_rows, variance)
    sigma = np.sqrt(variance)
    gaussian_matrix = np.random.normal(0, sigma, size=(n_rows, n_columns))
    return Matrix(n_rows, n_columns, gaussian_matrix)


def identity(size):
    return Matrix(size, size, np.identity(size))


def zero_matrix(n_rows, n_columns):
    return Matrix(n_rows, n_columns)


def symmetric_matrix_from_array(array):
    n = len(array)
    for i in range(n):
        for j in range(i + 1, n):
            array[j][i] = array[i][j]
    return Matrix(n, n, array)


def symmetric_matrix_to_array(matrix):
    n, m = matrix.dimensions()
    if n != m:
        raise ValueError("Matrix must be square to convert to symmetric array.")
    array = matrix.to_array()
    for i in range(n):
        for j in range(i + 1, n):
            array[j][i] = array[i][j]
    return array
