# import os
from functools import partial

import pandas as pd

from aif360.datasets import StandardDataset

TOTAL_MAPPING = {"New Orleans": 1, "Tennessee": 1, "Arizona": 1, "Minnesota": 1,
                 "Cincinnati": 1, "Denver": 1, "San Diego": 1, "Green Bay": 1,
                 "Jacksonville": 1, "Chicago": 1, "Indianapolis": 1,
                 "Los Angeles": 1, "New York": 1, "Dallas": 1, "Tampa Bay": 1,
                 "Buffalo": 1, "Oakland": 1, "San Francisco": 1, "Cleveland": 1,
                 "Houston": 1, "Pittsburgh": 1, "Carolina": 1, "Miami": 1,
                 "Washington": 1, "Kansas City": 1, "Atlanta": 1, "Baltimore": 1,
                 "Philadelphia": 1, "Detroit": 1, "Seattle": 1, "New England": 2}

def default_preprocessing(df, label_name):
    size = df.shape[1]
    df.rename(columns={0: 'team', 1: 'age', 2: 'height', 3: 'weight',
                       4: 'years_experience', 5: 'position', size-1: 'score',
                       size-2: 'pred_label', size-3: 'true_label'}, inplace=True)
    df = df[df.age > -1]
    df.replace(TOTAL_MAPPING, inplace=True)
    df.rename(columns={label_name: 'label'}, inplace=True)
    return df

class NFLDataset(StandardDataset):
    def __init__(self, label_name='true_label'):
        # train_path = "~/Downloads/deep_team_breakout_train.csv"
        test_path = "~/Downloads/deep_team_breakout_test.csv"
        # train = pd.read_csv(train_path, header=None)
        test = pd.read_csv(test_path, header=None)
        # df = pd.concat([train, test], ignore_index=True)
        df = test

        super(NFLDataset, self).__init__(df=df, label_name='label',
            favorable_classes=[1], protected_attribute_names=['team'],
            privileged_classes=[[2]], scores_name='score',
            features_to_drop=['pred_label' if label_name == 'true_label'
                         else 'true_label'] + range(6, df.shape[1]-3),
            custom_preprocessing=partial(default_preprocessing,
                                         label_name=label_name))
