import os

import numpy as np

def read_input_dataset(file_path):
    """
    Reads an input dataset where each row is an entry pair (x_t, y_t).
    x_t is d-dimensional, and y_t is a real number.

    Args:
        file_path (str): Path to the input dataset file.

    Returns:
        tuple: A tuple (X, y) where:
            - X is a numpy array of shape (n, d) containing the x_t vectors.
            - y is a numpy array of shape (n,) containing the y_t values.
    """
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]  # All columns except the last are x_t
    y = data[:, -1]   # The last column


def write_input_dataset(file_path, X, y):
    """
    Writes an input dataset to a file where each row is an entry pair (x_t, y_t).
    x_t is d-dimensional, and y_t is a real number.

    Args:
        file_path (str): Path to the output dataset file.
        X (numpy array): A numpy array of shape (n, d) containing the x_t vectors.
        y (numpy array): A numpy array of shape (n,) containing the y_t values.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data = np.hstack((X, y.reshape(-1, 1)))  # Combine X and y into a single array
    np.savetxt(file_path, data, delimiter=',')
