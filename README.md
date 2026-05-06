# On Differentially Private Mechanisms for Linear Regression
## Experiment Data and Usage

For each one of the mentioned datasets, please use the corresponding reader function located in:
`src/test/reader.py`

## Raw Data Downloads

For ACSIncome the function is self contained. For the other two datasets, please download the raw data files from the direct links below:

*   **Year**
    *   [Download YearPredictionMSD.zip](https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip)
*   **Household**
    *   [Download household_power_consumption.zip](https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip)

## Running the Experiments
To run the experiment for the algorithms mentioned in the paper, run `main.py`
To draw the plots for all of the mentioned datasets and to get the raw data of the main table of the experiments section, run `draw_all_plots.py`


## Repository Structure
```text
src/
├── algorithms/                 # Core algorithm implementations
│   ├── AdaSSP.py               
│   ├── Continual_AdaSSP.py     
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
   
