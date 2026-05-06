import math
from os import read

from src.judge.Judge import *
from src.test.reader import StreamingDataset


dataset_name = input()
print("dataset_name:", dataset_name)
dir = input()
print("dir:", dir)
input_file = input()
print("input_file:", input_file)
run_id = input()
print("run_id:", run_id)
eps = float(input())
print("eps:", eps)
snapshot_interval = int(input())

experiment_name = f"{dataset_name}_eps{eps}_run{run_id}"

reader = StreamingDataset(f"{dir}/{input_file}")

delta = 1.0 / reader.n
evaluate_continual(eps, delta, experiment_name, reader, snapshot_interval=snapshot_interval, output_path=f"{dir}/logs/")
