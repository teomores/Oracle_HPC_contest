import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import sys
from xgb_dataset_generation import *
import os

dir_path = ['validation',
            'validation_2',
            'validation_3',
            ]

for d in dir_path:
    path = os.path.join('dataset', d)
    if os.path.isfile(os.path.join(path, 'train_complete.csv')):
        df_val = pd.read_csv("dataset/validation/train_complete.csv")
    else:
        val = base_expanded_df(isValidation=True, save=False, path=path)
        val = adding_features(val, isValidation=True, path=path)
        save_path = os.path.join(path, "train_complete.csv")
        val.to_csv(save_path, index=False)
