from pathlib import Path

from src.test.reader import *

exp_names = ["year", "household", "ACSIncome"]
for exp in exp_names:
    Path(f"files/data/{exp}/logs/").mkdir(parents=True, exist_ok=True)

n, d, raw_X, raw_y, X_bound, y_bound = load_year_prediction_simple()
write_dataset("files/data/year/year.txt", n, d, raw_X, raw_y, X_bound, y_bound)

n, d, raw_X, raw_y, X_bound, y_bound = load_household_power()
write_dataset("files/data/household/household.txt", n, d, raw_X, raw_y, X_bound, y_bound)

load_acs_income_ca_2018(output_file="files/data/ACSIncome/ACSIncome.txt")
