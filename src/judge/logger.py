import matplotlib.pyplot as plt


class OutputLogger:
    def __init__(self, exp_name, curve_names):
        self.times = []
        self.points = []
        self.curve_names = curve_names
        self.exp_name = exp_name

    def add_point(self, t, point):
        self.times.append(t)
        self.points.append(point)

    def prepare(self):
        transposed_points = list(zip(*self.points))

        for i, curve in enumerate(transposed_points):
            label = self.curve_names[i] if i < len(self.curve_names) else f"Point {i+1}"
            plt.plot(self.times, curve, label=label)

        plt.xlabel("Time")
        plt.ylabel("F")
        plt.title(self.exp_name)
        plt.legend(loc="upper left")
        plt.yscale("log")

    def save(self):
        self.prepare()
        plt.savefig("files/plots/" + self.exp_name + "_plot.png")

    def show(self):
        self.prepare()
        plt.show()

    def print_to_file(self, exp_name):
        with open("files/logs/" + exp_name + "_log.txt", "w") as f:
            header = "time," + ",".join(
                self.curve_names[i] if i < len(self.curve_names) else f"Point {i+1}" for i in range(len(self.points[0]))
            ) + "\n"
            f.write(header)
            for t, point in zip(self.times, self.points):
                line = str(t) + "," + ",".join(map(str, point)) + "\n"
                f.write(line)

    def read_from_file(self, exp_name):
        self.times = []
        self.points = []
        with open("files/logs/" + exp_name + "_log.txt", "r") as f:
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split(",")
                t = int(parts[0])
                point = list(map(float, parts[1:]))
                self.times.append(t)
                self.points.append(point)


