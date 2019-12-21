import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import sys
from xgb_dataset_generation import *
import os

dir_path = [#'validation',
            #'validation_2',
            'validation_3',
            ]

for d in dir_path:
    print(f'Creating for {d}')
    path = os.path.join('dataset', d)
    if os.path.isfile(os.path.join(path, 'train_complete.csv')):
        df_val = pd.read_csv("dataset/validation/train_complete.csv")
    else:
        val = base_expanded_df(isValidation=True, save=True, path=path)
        #val = pd.read_csv(f"dataset/{d}/expanded/base_expanded_train.csv")
        val = adding_features(val, isValidation=True, path=path)
        save_path = os.path.join(path, "train_complete.csv")
        val.to_csv(save_path, index=False)
