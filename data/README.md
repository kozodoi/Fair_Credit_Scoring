# Data sets

The experiments are performed on seven credit scoring data sets:
- `german` - available at [the UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)).
- `taiwan` - available at [the UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).
- `pakdd` - available from [PAKDD 2010 Data Mining Competition](https://www.kdnuggets.com/2010/03/f-pakdd-2010-data-mining-competition.html).
- `gmsc` - available from [Kaggle Give Me Some Credit competition](https://www.kaggle.com/c/givemesomecredit]).
- `homecredit` - available from [Kaggle Home Credit Default Risk competition](https://www.kaggle.com/c/home-credit-default-risk).
- `bene` - obtained from: *Baesens, B., Van Gestel, T., Viaene, S., Stepanova, M., Suykens, J., & Vanthienen, J. (2003). Benchmarking state-of-the-art classification algorithms for credit scoring. Journal of the operational research society, 54(6), 627-635.*
- `uk` - obtained from: *Baesens, B., Van Gestel, T., Viaene, S., Stepanova, M., Suykens, J., & Vanthienen, J. (2003). Benchmarking state-of-the-art classification algorithms for credit scoring. Journal of the operational research society, 54(6), 627-635.*

The raw datasets have undergone some basic preprocessing, including correction of the feature names, removing redundant features and treating missing values.


# Structure

The data folder contains two subfolders:
- `raw/`: raw data sets in the CSV format
- `prepared/`: processed and partitioned data sets exported by the corresponding notebooks


# Instructions

Some of the raw data sets are not included in this repo. To perform the analysis on them, please:
1. Download the data sets using the links provided above.
2. Put the downloaded CSV files in the  `data/raw/` folder.
3. Make sure the CSV files are named according to the data set names provided above or adjust the CSV names in the codes.
4. Proceed with the steps outlined in [this README](https://github.com/kozodoi/Fair_Credit_Scoring/blob/main/README.md). 

Data sets `bene` and `uk` are subject to an NDA. Please contact the authors to request access to the data.
