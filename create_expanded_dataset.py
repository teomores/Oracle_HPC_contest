import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import sys
from xgb_dataset_generation import *
import os

# dir_path = ['validation',
#             'validation_2',
#             'validation_3',
#             ]
#
# for d in dir_path:
#     print(f'Creating for {d}')
#     path = os.path.join('dataset', d)
#     if os.path.isfile(os.path.join(path, 'train_complete.csv')):
#         # df_val = pd.read_csv("dataset/validation/train_complete.csv")
#         continue
#     elif os.path.isfile(os.path.join(path, 'expanded', 'base_expanded_train.csv')):
#         print("Load base_expanded_train")
#         val = pd.read_csv(os.path.join(path, 'expanded', 'base_expanded_train.csv'))
#         val = adding_features(val, isValidation=True, path=path)
#         save_path = os.path.join(path, "train_complete.csv")
#         val.to_csv(save_path, index=False)
#     else:
#         val = base_expanded_df(isValidation=True, save=False, path=path)
#         #val = pd.read_csv(f"dataset/{d}/expanded/base_expanded_train.csv")
#         val = adding_features(val, isValidation=True, path=path)
#         save_path = os.path.join(path, "train_complete.csv")
#         val.to_csv(save_path, index=False)

path_original = os.path.join('dataset','original')
if os.path.isfile(os.path.join('dataset/expanded', 'base_expanded_test.csv')):
    print("Load base_expanded_test")
    base_exp_test = pd.read_csv(os.path.join('dataset/expanded', 'base_expanded_test.csv'))
    base_exp_test = adding_features(base_exp_test, isValidation=False, path=path_original)
    save_path = os.path.join(path_original, 'test_complete.csv')
    base_exp_test.to_csv(save_path, index=False)
else:
    base_exp_test = base_expanded_df(isValidation=False, save=True, path=path_original)
    base_exp_test = adding_features(base_exp_test, isValidation=False, path=path_original)
    save_path = os.path.join(path_original, 'test_complete.csv')
    base_exp_test.to_csv(save_path, index=False)
