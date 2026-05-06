import numpy as np


class Matrix:
    def __init__(self, n_rows, n_columns, values=None):
        if values is None:
            self.data = np.zeros((n_rows, n_columns), dtype=np.float64)
        else:
            self.data = np.asarray(values, dtype=np.float64)
        self.n_rows, self.n_columns = self.data.shape

    def set_row(self, row_index, row_values):
        if row_values.n_columns != self.n_columns or row_values.n_rows != 1:
            raise ValueError("Length of row_values must match the number of columns.")
        for i in range(self.n_columns):
            self.data[row_index][i] = row_values.data[0][i]

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
        # if self.min_eigenvalue() < 1e-8:
        #     raise ValueError("Matrix is singular or nearly singular; cannot compute inverse.")
        return Matrix(self.n_rows, self.n_columns, np.linalg.pinv(self.data))

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
    random_matrix = []
    for i in range(n):
        random_matrix.append([])
        for j in range(n):
            random_matrix[i].append(np.float64(np.random.normal(0, sigma)))

    for i in range(n):
        for j in range(i + 1, n):
            random_matrix[j][i] = random_matrix[i][j]
    return Matrix(n, n, random_matrix)


def generate_gaussian_matrix(n_rows, n_columns, variance):
    if n_rows == n_columns:
        return generate_symmetric_gaussian_matrix(n_rows, variance)
    sigma = np.sqrt(variance)
    random_matrix = []
    for i in range(n_rows):
        random_matrix.append([])
        for j in range(n_columns):
            random_matrix[i].append(np.float64(np.random.normal(0, sigma)))

    return Matrix(n_rows, n_columns, random_matrix)


def identity(size):
    return Matrix(size, size, np.eye(size, dtype=np.float64))


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


def solve_regression(X, y):
    X_array = np.asarray(X.data, dtype=np.float64)
    y_array = np.asarray(y.data, dtype=np.float64).reshape(-1)

    theta, residuals, rank, s = np.linalg.lstsq(X_array, y_array, rcond=None)

    return Matrix(X.n_columns, y.n_columns, theta.reshape(-1, 1))
