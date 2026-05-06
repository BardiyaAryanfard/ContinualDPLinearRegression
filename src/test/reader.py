import numpy as np
from pathlib import Path


class StreamingDataset:
    def __init__(self, filename):
        self._file = open(filename, "r")
        header = self._file.readline().split()
        self.n = int(header[0])
        self.d = int(header[1])
        self.X_bound = float(header[2])
        self.y_bound = float(header[3])
        self._read = 0

    def next_data_point(self):
        if self._read >= self.n:
            raise StopIteration("No more data points")
        line = self._file.readline().split()
        x = np.array([float(v) for v in line[:self.d]]).reshape(1, -1)
        y = float(line[self.d])
        self._read += 1
        return x, y

    def close(self):
        if not self._file.closed:
            self._file.close()

    def __del__(self):
        self.close()


def write_dataset(filename, n, d, X, y, X_bound, y_bound):
    with open(filename, "w") as f:
        f.write(f"{n} {d} {X_bound} {y_bound}\n")
        for i in range(n):
            row = " ".join(f"{v:.8g}" for v in X[i])
            f.write(f"{row} {y[i]:.8g}\n")


def load_year_prediction_simple(X_bound=1.0, y_bound=1.0):
    data = np.loadtxt("files/data/year/YearPredictionMSD.txt", delimiter=",")
    y_raw = data[:, 0]
    X_raw = data[:, 1:13]
    feat_bound = 100.0
    X = X_bound * X_raw / feat_bound
    X = np.clip(X, -X_bound, X_bound)
    X = X / np.sqrt(X.shape[1])

    year_min, year_max = 1922.0, 2011.0
    y = 2 * y_bound * (y_raw - year_min) / (year_max - year_min) - y_bound
    y = np.clip(y, -y_bound, y_bound)

    n = len(y)
    d = X.shape[1]

    return n, d, X, y, X_bound, y_bound


def load_household_power(X_bound=1.0, y_bound=1.0):
    data_file = Path("files") / "data" / "household" / "household_power_consumption.txt"

    raw = np.genfromtxt(data_file, delimiter=";", skip_header=1, usecols=(2, 3, 4, 5, 6, 7, 8))
    mask = np.all(np.isfinite(raw), axis=1)
    raw = raw[mask]

    y_raw = raw[:, 0]
    X_raw = raw[:, 1:]

    feat_min = np.array([0.0, 220.0, 0.0, 0.0, 0.0, 0.0])
    feat_max = np.array([1.4, 255.0, 48.0, 88.0, 80.0, 31.0])

    X = 2 * X_bound * (X_raw - feat_min) / (feat_max - feat_min) - X_bound
    X = np.clip(X, -X_bound, X_bound)
    X = X / np.sqrt(X.shape[1])

    y = 2 * y_bound * (y_raw - 0.0) / (11.0 - 0.0) - y_bound
    y = np.clip(y, -y_bound, y_bound)

    n = len(y)
    d = X.shape[1]
    return n, d, X, y, X_bound, y_bound


def load_acs_income_ca_2018(X_bound=1.0, y_bound=1.0, output_file="files/data/ACSIncome/ACSIncome_ca_new.txt"):
    from folktables import ACSDataSource, adult_filter

    data_source = ACSDataSource(survey_year="2018",
                                horizon="1-Year",
                                survey="person")
    data = data_source.get_data(states=["CA"], download=True)

    data = adult_filter(data)

    cols = ["AGEP", "SCHL", "WKHP", "SEX", "PINCP"]
    data = data[cols].dropna()

    X_raw = data[cols[:-1]].values.astype(float)
    y_raw = data["PINCP"].values.astype(float)

    feat_min = np.array([16.0, 1.0, 1.0, 1.0])
    feat_max = np.array([99.0, 24.0, 99.0, 2.0])

    X = 2 * X_bound * (X_raw - feat_min) / (feat_max - feat_min) - X_bound
    X = np.clip(X, -X_bound, X_bound)
    X = X / np.sqrt(X.shape[1])

    income_min = 100.0
    income_max = 500000.0

    y = 2 * y_bound * (y_raw - income_min) / (income_max - income_min) - y_bound
    y = np.clip(y, -y_bound, y_bound)

    perm = np.random.default_rng(42).permutation(len(y))
    X, y = X[perm], y[perm]
    n = len(y)
    d = X.shape[1]

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(f"{n} {d} {X_bound} {y_bound}\n")
        for i in range(n):
            row = " ".join(f"{v:.8g}" for v in X[i])
            f.write(f"{row} {y[i]:.8g}\n")

    print(f"Wrote {output_file}: n={n}, d={d}, X_bound={X_bound}, y_bound={y_bound}")
    return n, d, X, y, X_bound, y_bound

