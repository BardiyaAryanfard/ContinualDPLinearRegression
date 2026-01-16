import numpy as np

def read_data_file(file_path, y_column=-1, delimiter=None):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if delimiter is None:
        delimiter = ',' if ',' in lines[0] else None

    n = len(lines)
    d = len(lines[0].strip().split(delimiter)) - 1

    X = np.zeros((n, d))
    y = np.zeros((n, 1))

    for i, line in enumerate(lines):
        data = list(map(float, line.strip().split(delimiter)))
        if y_column == -1:
            X[i] = data[:d]
            y[i] = data[d]
        else:
            X[i] = data[:y_column] + data[y_column + 1:]
            y[i] = data[y_column]

    return n, d, X, y.flatten()