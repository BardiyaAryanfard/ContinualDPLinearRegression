# On Differentially Private Mechanisms for Linear Regression
This repository contains the code for the experiments of the paper "On Differentially Private Mechanisms for Linear Regression" (NeurIPS 2026) by Bardiya Aryanfard, Monika Henzinger and Farhood Rostamkhani.

## Requirements
Python 3.12 with numpy, matplotlib and folktables:
```
pip install numpy matplotlib folktables
```
All of the scripts should be run from the root of the repository.

## Data
ACSIncome is downloaded automatically using folktables. For the other two datasets, please download the raw data files from the links below and unzip them into the given folders:

*   **Year**
    *   [Download YearPredictionMSD.zip](https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip) and put `YearPredictionMSD.txt` in `files/data/year/`
*   **Household**
    *   [Download household_power_consumption.zip](https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip) and put `household_power_consumption.txt` in `files/data/household/`

Then run:
```
python write_all_datasets.py
```
This writes the input files `files/data/year/year.txt`, `files/data/household/household.txt` and `files/data/ACSIncome/ACSIncome.txt`, and creates the `logs/` folder for each dataset.

## Running the Experiments
`main.py` runs all of the algorithms mentioned in the paper (AdaSSP with c = 1/16 and c = 1, PQFTL, Batch, non-private and trivial) on one dataset with one value of ε. It reads 6 lines from the standard input:
1. dataset name (`year`, `household` or `ACSIncome`)
2. dataset folder (e.g. `files/data/year`)
3. input file name (e.g. `year.txt`)
4. run id
5. ε
6. snapshot interval S (the loss function is calculated every S time steps)

For example, for the first run on the Year dataset with ε = 1, make a file `input.txt`:
```
year
files/data/year
year.txt
1
1.0
<S>
```
and run:
```
python main.py < input.txt
```
The loss function of every algorithm at every snapshot, together with the running times, is written to `files/data/year/logs/year_eps1.0_run1_log.txt`.

To reproduce the results of the paper, run this for all three datasets, ε = 1.0 and 2.0, and run ids 1 to 10. Then run:
```
python draw_all_plots.py
```
to draw the plots for all of the datasets and to get the rows of the main table of the experiments section (median and standard deviation over the 10 runs).

### Synthetic Datasets
The synthetic datasets of Appendix I are generated with `generator_heuristic` in `src/test/gen_synthetic_datasets.py` (the parameters of the 12 datasets used for choosing c are listed in that file) and can be written to an input file with `write_dataset` in `src/test/reader.py`. Then run `main.py` on them as above. To try other values of c, change `c_lambdas` and `c_lambdas_str` in `src/judge/Judge.py`.

## Repository Structure
```text
main.py                         # Runs one experiment
write_all_datasets.py           # Prepares the input files of the real-world datasets
draw_all_plots.py               # Plots and main table of the experiments section
src/
├── algorithms/                 # Core algorithm implementations
│   ├── AdaSSP.py               # Batch AdaSSP (Algorithm 3 and Algorithm 4)
│   ├── Continual_AdaSSP.py     # ContinualAdaSSP (Algorithm 1)
│   └── PQFTL.py
├── judge/
│   ├── Judge.py                # Main evaluation function
│   └── logger.py               # Handles experiment logging and output
├── test/                       # Datasets
│   ├── gen_synthetic_datasets.py # Generates synthetic data for heuristic improvement
│   └── reader.py               # Data loaders for all datasets
└── util/                       # Helper classes and data structures
    ├── BinaryTree.py
    ├── Interval.py
    ├── Matrix.py
    └── SVT.py
```

## Citation
```bibtex
@inproceedings{aryanfard2026differentially,
  title     = {On Differentially Private Mechanisms for Linear Regression},
  author    = {Aryanfard, Bardiya and Henzinger, Monika and Rostamkhani, Farhood},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License
This code is released under the Apache License 2.0 (see `LICENSE`).

The Year and Household datasets are from the UCI Machine Learning Repository and are licensed under CC BY 4.0. ACSIncome is built from the ACS PUMS data of the U.S. Census Bureau and is downloaded using folktables; its use is subject to the terms of service of the Census Bureau.
