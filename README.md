# Fair Credit Scoring

![pipeline](/output/fig_pipeline.jpg)

Codes and data accompanying the paper:

`
Kozodoi, N., Jacob, J., Lessmann, S. (2020). Fairness in Credit Scoring: Assessment, Implementation and Profit Implications (submitted to European Journal of Operational Research)
`


## Summary

The rise of algorithmic decision-making has spawned much research on fair machine learning (ML). Financial institutions use ML for building risk scorecards that support a range of credit-related decisions. Yet, the literature on fair ML in credit scoring is scarce. The paper makes three contributions:
1. We revisit statistical fairness criteria and examine their adequacy for credit scoring.
2. We catalog algorithmic options for incorporating fairness goals in the ML model development pipeline.
3. We empirically compare different fairness processors in a profit-oriented credit scoring context using real-world data.

The empirical results substantiate the evaluation of fairness measures, identify more and less suitable options to implement fair credit scoring, and clarify the profit-fairness trade-off in lending decisions. Specifically, we find that multiple fairness criteria can be approximately satisfied at once and identify separation as a proper criterion for measuring the fairness of a scorecard. We also find fair in-processors to deliver a good balance between profit and fairness. More generally, we show that algorithmic discrimination can be reduced to a reasonable level at a relatively low cost.

This repo contains codes, files and data sets that allow reproducing the results presented in the paper, performing additional analyses and extending the experimental setup. The codes provide implementations of eight fairness processors on seven credit scoring data sets. Further details on the modeling pipeline are provided in the paper as well as in the codes.


## Repo structure

The repo has the following structure:
- `codes/`: Python notebooks and R codes implementing data processing, fairness algorithms and analysis of the results.
- `data/`: input data, including the raw data sets in CSV and prepared data exported from the data processing scripts.
- `functions/`: helper functions accompanying the codes, including a customized `aif360` Python module.
- `output/`: figures and tables with empirical results exported from `code_14_results.R` and presented in the paper.
- `results/`: intermediate and final results files exported from the different modeling codes.

There are 15 code files:
- `code_00_working_paths.R`: sets working paths for the project subfolders.
- `code_01_data_partitioning.ipynb`: partitions and preprocesses a data set and exports prepared data.
- `code_02_preprocess1.ipynb`: implements reweighing and disparate impact remover and exports processed data.
- `code_03_preprocess2.R`: implements base classifiers on fair pre-processed data.
- `code_04_preprocess3.R`: performs meta-parameter tuning of disparate impact remover.
- `code_05_inprocess1.ipynb`: implements prejudice remover and meta fair algorithm.
- `code_06_inprocess2.ipynb`: implements adversarial debiasing.
- `code_07_inprocess3.R`: performs meta-parameter tuning of fair in-processors.
- `code_08_postprocess1.R`: implements base classifiers on the original data.
- `code_09_postprocess2.R`: implements equalized odds processor.
- `code_10_postprocess3.R`: implements group-wise Platt scaling.
- `code_11_postprocess4.ipynb`: implements reject option classification.
- `code_12_postprocess5.R`: processes reject option classification results.
- `code_13_unconstrained.R`s: implements an unconstrained profit maximization benchmark.
- `code_14_results.R`: performs aggregation and analysis of the experimental results.

Further details and documentation on codes and helper functions are provided in the corresponding files.


## Working with the repo

### Dependencies

The codes require Python 3.7+ and R 4.0+. Relevant R packages are installed separately at the beginning of each R script. Packages required to implement each of the considered fairness processors are listed and imported in the corresponding Python scripts.

To work with the Python notebooks, we recommend to create a virtual Conda environment from the supplied `environment.yml` file:
```
conda env create --name fairness --file environment.yml
conda activate fairness
```

### Reproducing results

Reproducing results presented in the paper and carrying out additional analysis involves the following steps:
1. Update working paths in `code_00_working_paths.R`.
2. Execute `code_14_results.R` to perform analysis of the results and save outputs in `output/`.


### Extending experimental setup

Modifying the experimental setup requires the following:
1. Make sure the raw data sets are located in the `data/raw/` folder.
2. Update working paths in `code_00_working_paths.R`.
3. Execute `code_01_data_partitioning.ipynb` to process and partition each data set.
4. Adjust experimental parameters specified in `functions/`.
5. Execute the modeling codes according to their name order for each of the data sets.
6. Execute `code_14_results.R` to perform analysis of the results and save outputs in `output/`.


## Data sets

The experiments are performed on seven credit scoring data sets:
- `german` - available at [the UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)).
- `taiwan` - available at [the UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).
- `pakdd` - availbale from [PAKDD 2010 Data Mining Competition](https://www.kdnuggets.com/2010/03/f-pakdd-2010-data-mining-competition.html).
- `gmsc` - availbale from [Kaggle Give Me Some Credit competition](https://kaggle.com/c/givemesomecredit]).
- `homecredit` - availbale from [Kaggle Home Credit Default Risk competition](https://kaggle.com/c/home-credit-default-risk).
- `bene` - obtained from: *Baesens, B., Van Gestel, T., Viaene, S., Stepanova, M., Suykens, J., & Vanthienen, J. (2003). Benchmarking state-of-the-art classification algorithms for credit scoring. Journal of the operational research society, 54(6), 627-635.*
- `uk` - obtained from: *Baesens, B., Van Gestel, T., Viaene, S., Stepanova, M., Suykens, J., & Vanthienen, J. (2003). Benchmarking state-of-the-art classification algorithms for credit scoring. Journal of the operational research society, 54(6), 627-635.*

The raw data sets `bene` and `uk` are subject to an NDA and are not included in this repo.
