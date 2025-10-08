from src.util.Interval import Interval
from src.util.Matrix import Matrix


class BCMOpt:
    def __init__(self, T, epsilon, n_rows, n_columns, noise_variance):
        self.root = Node(None, Interval(0, T), T, epsilon, True, n_rows, n_columns, noise_variance)
        self.T = T
        self.epsilon = epsilon

    def update(self, value):
        return self.root.update(value)

    def get(self):
        return self.root.last

    def get_real_sum(self):
        return self.root.value


class Node:
    # noise_variance is the variance of the Gaussian noise to be added to each entry of the matrix at each node
    def __init__(self, father, interval, T, epsilon, is_left, n_rows, n_columns, noise_variance):
        self.T = T
        self.epsilon = epsilon
        self.father = father
        self.interval = interval
        self.noise = generate_gaussian_matrix(n_rows, n_columns, noise_variance)
        self.value = Matrix(n_rows, n_columns)
        self.is_full = False
        self.noisy_left_sum = Matrix(n_rows, n_columns)
        self.is_left_child_full = False
        self.last = Matrix(n_rows, n_columns)
        self.is_left = is_left
        if interval.len() > 1:
            self.child = Node(self, interval.left_half(), T, epsilon, True, n_rows, n_columns, noise_variance)

    def update(self, value):
        self.value += value
        if self.interval.len() == 1:
            self.is_full = True
            self.last = self.value.add(self.noise)
            return self.value.add(self.noise)
        self.last = self.child.update(value).add(self.noisy_left_sum)
        if self.child.is_full:
            if self.child.is_left:
                self.noisy_left_sum = self.child.last
                self.child = Node(self, self.interval.right_half(), self.T, self.epsilon, False, self.value.n_rows,
                                  self.value.n_columns, self.noise.variance)
            else:
                self.last = self.value.add(self.noise)
                self.is_full = True
        return self.last
