import math

import numpy as np


class SVT:
    def __init__(self, epsilon, delta, n, sensitivity, c):
        self.epsilon = epsilon
        self.delta = delta
        self.n = n
        self.sensitivity = sensitivity
        self.c = c
        self.beta = 0.1
        self.sigma = math.sqrt(32.0 * c * math.log(1.0 / delta)) * sensitivity / epsilon
        print("sigma: ", self.sigma)
        self.rho = np.random.laplace(0.0, self.sigma)
        self.count = 0

    def update(self, query_value, threshold):
        if self.count >= self.c:
            if self.count == self.c:
                print("SVT: Exceeded maximum number of positive responses T")
            self.count += 1
            return False

        noisy_query = query_value + np.random.laplace(0.0, 2.0 * self.sigma)
        noisy_threshold = threshold + self.rho
        if noisy_query >= noisy_threshold:
            self.count += 1
            return True
        else:
            return False

    def get_alpha(self):
        return self.sensitivity * (math.log(self.n) + math.log(2.0 * self.c / self.beta)) * math.sqrt(
            512.0 * self.c * math.log(1.0 / self.delta)) / self.epsilon

