from src.judge.logger import AggregatedLogger

exp_names = ["year", "household", "ACSIncome"]
for exp in exp_names:
    for eps in ["1.0", "2.0"]:
        exp_name = f"{exp}_eps{eps}"
        path = f"files/data/{exp}/logs/"
        agg_logger = AggregatedLogger(exp_name=exp_name)
        runs = [f"{exp}_eps{eps}_run{i}" for i in range(1, 11)]
        agg_logger.read_multiple_files(exp_names=runs, path=path, selected_curves=["non-private", "trivial", "Batch", "PQFTL"] + [f"AdaSSP{c}" for c in ["1/16", "1"]])
        if exp == "year" and eps == "1.0":
            print(str(agg_logger.curve_names))
        agg_logger.show()
        agg_logger.print_table_row(dataset_label=exp, eps=eps, tail_fraction=1.0)