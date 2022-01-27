# Fair ML in Credit Scoring

![pipeline](/output/fig_pipeline.jpg)

Code and results accompanying the paper:

```
Kozodoi, N., Jacob, J., Lessmann, S. (2021).
Fairness in Credit Scoring: Assessment, Implementation and Profit Implications.
European Journal of Operational Research, 297, 1083-1094.
``` 

The paper is available at the [publisher's website](https://doi.org/10.1016/j.ejor.2021.06.023). The preprint is also available [on ArXiV](https://arxiv.org/abs/2103.01907).

- [Summary](#summary)
- [Repo structure](#repo-structure)
- [Data sets](#data-sets)
- [Working with the repo](#working-with-the-repo)


## Summary

The rise of algorithmic decision-making has spawned much research on fair machine learning (ML). Financial institutions use ML for building risk scorecards that support a range of credit-related decisions. The paper makes three contributions:
1. We revisit statistical fairness criteria and examine their adequacy for credit scoring.
2. We catalog algorithmic options for incorporating fairness goals in the ML model development pipeline. 
3. We empirically compare different fairness processors in a profit-oriented credit scoring context using real-world data. 

The empirical results substantiate the evaluation of fairness measures, identify suitable options to implement fair credit scoring, and clarify the profit-fairness trade-off in lending decisions. We find that multiple fairness criteria can be approximately satisfied at once and recommend separation as a proper criterion for measuring scorecard fairness. We also find fair in-processors to deliver a good profit-fairness balance and show that algorithmic discrimination can be reduced to a reasonable level at a relatively low cost. 

This repo contains codes and files that allow reproducing the results presented in the paper, performing additional analyses and extending the experimental setup. We provide implementations of eight fairness processors used in the paper. Further details on the modeling pipeline are provided in the paper as well as in the codes.


## Repo structure

The repo has the following structure:
- `codes/`: Python notebooks and R codes implementing data processing and fair ML algorithms
- `data/`: raw data sets and processed data exported from the data processing scripts
- `functions/`: helper functions accompanying the codes, including a customized `aif360` Python module
- `output/`: figures and tables with results exported from `code_14_results.R` and presented in the paper
- `results/`: intermediate and final results files exported from the different modeling codes

Further details on the code files are provided in the [README file](https://github.com/kozodoi/Fair_Credit_Scoring/blob/main/codes/README.md) in the codes folder.


## Data sets

The experiments are performed on seven credit scoring data sets obtained from different sources, including the UCI Machine Learning Repository, Kaggle and PAKDD platforms. 

To respect the terms of use of the corresponding data sets and adhere to the file size limit, the raw data is not included in the repo. The full list of data sets with the download links and instructions is provided in the [README file](https://github.com/kozodoi/Fair_Credit_Scoring/blob/main/data/README.md) in the data folder.


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
1. Place the raw data sets in the `data/raw/` folder (see [instructions](https://github.com/kozodoi/Fair_Credit_Scoring/blob/main/data/README.md)).
2. Update working paths in `code_00_working_paths.R`.
3. Execute `code_01_data_partitioning.ipynb` to process and partition each data set.
4. Adjust experimental parameters specified in `functions/`.
5. Execute the modeling codes according to their name order for each of the data sets.
6. Execute `code_14_results.R` to perform analysis of the results and save outputs in `output/`.
