class Interval:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def len(self):
        return self.right - self.left

    def left_half(self):
        return Interval(self.left, int((self.left + self.right) / 2))

    def right_half(self):
        return Interval(int((self.left + self.right) / 2), self.right)

    def contains(self, index):
        return (self.left <= index) and (index < self.right)

    def contains_interval(self, interval):
        return (self.left <= interval.left) and (interval.right <= self.right)
