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
- [Fair ML methods](#fair-ml-methods)
- [Working with the repo](#working-with-the-repo)


## Summary

The rise of algorithmic decision-making has spawned much research on fair machine learning (ML). Financial institutions use ML for building risk scorecards that support a range of credit-related decisions. The project makes three contributions:
1. Revisiting statistical fairness criteria and examine their adequacy for credit scoring.
2. Cataloging algorithmic options for incorporating fairness goals in the ML model development pipeline. 
3. Empirically comparing multiple fairness processors in a profit-oriented credit scoring context using real-world data. 

The empirical results substantiate the evaluation of fairness measures, identify suitable options to implement fair credit scoring, and clarify the profit-fairness trade-off in lending decisions. We find that multiple fairness criteria can be approximately satisfied at once and recommend separation as a proper criterion for measuring scorecard fairness. We also find fair in-processors to deliver a good profit-fairness balance and show that algorithmic discrimination can be reduced to a reasonable level at a relatively low cost. 

![results](https://i.postimg.cc/5yB7y21M/fair-gif.gif)

This repo contains codes and files that allow reproducing the results presented in the paper, performing additional analyses and extending the experimental setup. It provides implementations of eight fairness processors used in the paper. Further details on the modeling pipeline are provided in the paper as well as in the codes.


## Repo structure

The repo has the following structure:
- `codes/`: Python notebooks and R codes implementing data processing and fair ML algorithms
- `data/`: raw data sets and processed data exported from the data processing scripts
- `functions/`: helper functions accompanying the codes, including a customized `aif360` Python module
- `output/`: figures and tables with results exported from `code_14_results.R` and presented in the paper
- `results/`: intermediate and final results files exported from the different modeling codes

Further details on the codes are provided in a separate [README file](https://github.com/kozodoi/Fair_Credit_Scoring/blob/main/codes/README.md) in the codes folder.


## Data sets

The experiments are performed on seven credit scoring data sets obtained from different sources, including the UCI Machine Learning Repository, Kaggle and PAKDD platforms. 

To respect the terms of use of the corresponding data sets and adhere to the file size limit, the raw data is not included in the repo. The full list of data sets with the download links and instructions is provided in a separate [README file](https://github.com/kozodoi/Fair_Credit_Scoring/blob/main/data/README.md) in the data folder.


## Fair ML methods

The paper implements and banchmarks the folloeing fair ML algorithms:
1. Pre-processors:
  - Reweighting [(Calders et al. 2009)](https://ieeexplore.ieee.org/abstract/document/5360534)
  - Disparate Impact Remover [(Feldman et al. 2015)](https://dl.acm.org/doi/abs/10.1145/2783258.2783311?casa_token=hPPsvh9w2QEAAAAA:RE90pNifv99Y9yCMgE4O1vOquljiAtjVCQQ3UgFDHIgcn2J21J5ry6HCv2iXXTX2Gw9e1VBbS07j)
2. In-processors:
  - Prejudice Remover [(Kamishima et al. 2012)](https://link.springer.com/chapter/10.1007/978-3-642-33486-3_3)
  - Adersarial Debiasing [(Zhang et al. 2018)](https://dl.acm.org/doi/abs/10.1145/3278721.3278779)
  - Meta-Fairness Algorithm [(Celis et al. 2019)](https://dl.acm.org/doi/abs/10.1145/3287560.3287586?casa_token=VdBhACPUHUYAAAAA:D8-vlR7Vf5QVQXyYhHB23IBjO0xrKQH64wztDghcSCUpaUwwkWeMZ2Cqu76yjLvSCVhzpjleAAnJ)
3. Post-processors:
  - Reject Option Classification [(Kamiran et al. 2012)](https://ieeexplore.ieee.org/abstract/document/6413831)
  - Equalized Odds Processor [(Hardt et al. 2016)](https://papers.nips.cc/paper/2016/hash/9d2682367c3935defcb1f9e247a97c0d-Abstract.html)
  - Platt Scaling [(Platt 1999)](https://home.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf), [(Barocas et al. 2019)](https://fairmlbook.org)


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
