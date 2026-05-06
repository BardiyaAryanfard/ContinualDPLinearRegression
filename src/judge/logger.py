import matplotlib.pyplot as plt
import numpy as np


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
        plt.clf()

        transposed_points = list(zip(*self.points))

        for i, curve in enumerate(transposed_points):
            label = self.curve_names[i] if i < len(self.curve_names) else f"Point {i + 1}"
            plt.plot(self.times, curve, label=label)

        plt.xlabel("Time")
        plt.ylabel("F")
        plt.title(self.exp_name)
        plt.legend(loc="lower right")
        plt.yscale("log")

    def save(self):
        self.prepare()
        plt.savefig("files/plots/" + self.exp_name + "_plot.png")

    def show(self):
        self.prepare()
        plt.show()

    def print_to_file(self, exp_name, path="files/logs/"):
        with open(path + exp_name + "_log.txt", "w") as f:
            header = "time," + ",".join(
                self.curve_names[i] if i < len(self.curve_names) else f"Point {i + 1}" for i in
                range(len(self.points[0]))
            ) + "\n"
            f.write(header)
            for t, point in zip(self.times, self.points):
                line = str(t) + "," + ",".join(map(str, point)) + "\n"
                f.write(line)

    def read_from_file(self, exp_name, path="files/logs/", selected_curves=None):
        self.times = []
        self.points = []
        self.exp_name = exp_name

        with open(path + exp_name + "_log.txt", "r") as f:
            lines = f.readlines()

            labels = lines[0].strip().split(",")
            all_curve_names = labels[1:]

            if selected_curves is None:
                selected_indices = list(range(len(all_curve_names)))
                self.curve_names = all_curve_names
            else:
                selected_indices = [
                    i for i, name in enumerate(all_curve_names)
                    if name in selected_curves
                ]
                self.curve_names = [all_curve_names[i] for i in selected_indices]

            lines = lines[1:]

            for line in lines:
                parts = line.strip().split(",")
                t = int(parts[0])

                values = list(map(float, parts[1:]))

                filtered_values = [values[i] for i in selected_indices]

                self.times.append(t)
                self.points.append(filtered_values)



class AggregatedLogger:
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.times = []
        self.curve_names = []
        self.all_points = []

    def print_table_row(self, dataset_label, eps, tail_fraction=1.0):

        n_steps = self.all_points.shape[1]
        tail_start = int((1.0 - tail_fraction) * n_steps)

        tail_per_run = np.mean(self.all_points[:, tail_start:, :], axis=1)

        medians = np.median(tail_per_run, axis=0)
        stds = np.std(tail_per_run, axis=0)

        cells = [dataset_label, str(eps)]
        for i, name in enumerate(self.curve_names):
            val = f"{medians[i]:.2g} ({stds[i]:.2g})"
            cells.append(val)

        row = " & ".join(cells) + r" \\"
        print(row)
        return row

    def read_multiple_files(self, exp_names, path="files/logs/", selected_curves=None):
        runs_data = []

        for name in exp_names:
            temp_logger = OutputLogger("", [])
            temp_logger.read_from_file(name, path=path, selected_curves=selected_curves)

            if not self.times:
                self.times = temp_logger.times
                self.curve_names = temp_logger.curve_names

            runs_data.append(temp_logger.points)

        self.all_points = np.array(runs_data)

    def _plot_aggregated_curves(self, data_array, ylabel, loc, lower_percentile=25, upper_percentile=75):
        plt.clf()

        COLORS = {
            "AdaSSP1/16": "#0072B2",
            "AdaSSP1": "#56B4E9",
            "PQFTL": "#E69F00",
            "Batch": "#CC0066",
            "non-private": "#00A14B",
            "trivial": "#000000",
        }

        medians = np.percentile(data_array, 50, axis=0)
        lowers = np.percentile(data_array, lower_percentile, axis=0)
        uppers = np.percentile(data_array, upper_percentile, axis=0)

        for i in range(medians.shape[1]):
            label = self.curve_names[i] if i < len(self.curve_names) else f"Point {i + 1}"
            color = COLORS.get(label, None)
            p = plt.plot(self.times, medians[:, i], label=label, color=color)
            color = p[0].get_color()

            plt.fill_between(self.times, lowers[:, i], uppers[:, i], color=color, alpha=0.2)

        plt.xlabel("Time")
        plt.ylabel(ylabel)
        plt.title(self.exp_name)
        plt.legend(loc=loc)
        plt.yscale("log")
        plt.show()

    def show(self, lower_percentile=25, upper_percentile=75):
        self._plot_aggregated_curves(
            self.all_points,
            ylabel="F",
            loc="upper left",
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile
        )
