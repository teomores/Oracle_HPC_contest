import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy import *
from scipy.sparse import *
from pathlib import Path

from features.target import target
import os
import time
import argparse

def base_expanded_df(alpha = 0.2, beta = 0.2, gamma = 0.2, k = 50, isValidation=False, save=True, path=""):

    sim_path = os.path.join(path, 'similarities')

    if isValidation:
        val_name = path.split("/")[-1]   # Windows
        #val_name = path.split("/")[-1]   # Mac
        print(val_name)
        train_path = os.path.join(path, 'train.csv')
        test_path = os.path.join(path, 'test.csv')
        print(test_path)
        df_train = pd.read_csv(train_path, escapechar="\\")
        df_test = pd.read_csv(test_path, escapechar="\\")

        sim_name = load_npz(os.path.join(sim_path, f'jaccard_uncleaned_name_300k_{val_name}_3ngrams.npz'))
        sim_email = load_npz(os.path.join(sim_path, f'jaccard_uncleaned_email_300k_{val_name}_2ngrams.npz'))
        sim_phone = load_npz(os.path.join(sim_path, f'jaccard_uncleaned_phone_300k_{val_name}_2ngrams.npz'))
        sim_address = load_npz(os.path.join(sim_path, f'jaccard_uncleaned_address_300k_{val_name}_2ngrams.npz'))
    else:
        sim_name = load_npz(os.path.join(sim_path, 'jaccard_uncleaned_name_300k_original_3ngrams.npz'))
        sim_email = load_npz(os.path.join(sim_path, 'jaccard_uncleaned_email_300k_original_2ngrams.npz'))
        sim_phone = load_npz(os.path.join(sim_path, 'jaccard_uncleaned_phone_300k_original_2ngrams.npz'))
        sim_address = load_npz(os.path.join(sim_path, 'jaccard_uncleaned_address_300k_original_2ngrams.npz'))
        df_train = pd.read_csv('dataset/original/train.csv', escapechar="\\")
        df_test = pd.read_csv('dataset/original/test.csv', escapechar="\\")

    hybrid = sim_name + alpha * sim_email + beta * sim_phone + gamma * sim_address

    df_train = df_train.sort_values(by=['record_id']).reset_index(drop=True)
    df_test = df_test.sort_values(by=['record_id']).reset_index(drop=True)

    linid_ = []
    linid_idx = []
    linid_score = []
    linid_name_cosine = []
    linid_email_cosine = []
    linid_phone_cosine = []
    linid_address_cosine = []
    linid_record_id = []


    tr = df_train[['record_id', 'linked_id']]
    for x in tqdm(range(df_test.shape[0])):
        #df = df_train.loc[hybrid[x].nonzero()[1][hybrid[x].data.argsort()[::-1]],:][:k]
        indices = hybrid[x].nonzero()[1][hybrid[x].data.argsort()[::-1]][:k]
        # # add phone top indexes
        # ind_phone = []
        # tol = 0.625
        # if np.argwhere(sim_phone[x]>=tol).shape[0]>0:
        #     ind_phone = np.argwhere(sim_phone[x]>=tol)[0][1:]
        #
        # indices = list(set(list(indices)+list(ind_phone)))
        df = tr.loc[indices, :][:k]
        linid_.append(df['linked_id'].values)
        linid_idx.append(df.index)
        linid_record_id.append(df.record_id.values)
        linid_score.append(np.sort(hybrid[x].data)[::-1][:k])
        linid_name_cosine.append([sim_name[x, t] for t in indices])
        linid_email_cosine.append([sim_email[x, t] for t in indices])
        linid_phone_cosine.append([sim_phone[x, t] for t in indices])
        linid_address_cosine.append([sim_phone[x,t] for t in indices])

    df = pd.DataFrame()
    df['queried_record_id'] = df_test.record_id
    df['predicted_record_id'] = linid_
    df['predicted_record_id_record'] = linid_record_id
    df['cosine_score'] = linid_score
    df['name_cosine'] = linid_name_cosine
    df['email_cosine'] = linid_email_cosine
    df['phone_cosine'] = linid_phone_cosine
    df['address_cosine'] = linid_address_cosine
    df['linked_id_idx'] = linid_idx

    df_new = expand_df(df)

    if save:
        if isValidation:
            if not os.path.isdir(os.path.join(path, "expanded")):
                os.makedirs(os.path.join(path, "expanded"))
            save_path = os.path.join(path, "expanded/base_expanded_train.csv")
            df_new.to_csv(save_path, index=False)
        else:
            df_new.to_csv("dataset/expanded/base_expanded_test.csv", index=False)

    return df_new



def expand_df(df):
    df_list = []
    for (q, pred, pred_rec, score, s_name, s_email, s_phone, s_addr,  idx) in tqdm(
            zip(df.queried_record_id, df.predicted_record_id, df.predicted_record_id_record, df.cosine_score,
                df.name_cosine, df.email_cosine, df.phone_cosine, df.address_cosine, df.linked_id_idx)):
        for x in range(len(pred)):
            df_list.append((q, pred[x], pred_rec[x], score[x], s_name[x], s_email[x], s_phone[x], s_addr[x],  idx[x]))

    # TODO da cambiare predicted_record_id in predicted_linked_id e 'predicted_record_id_record' in 'predicted_record_id'
    df_new = pd.DataFrame(df_list, columns=['queried_record_id', 'predicted_record_id', 'predicted_record_id_record',
                                            'cosine_score', 'name_cosine',
                                            'email_cosine', 'phone_cosine', 'address_cosine', 'linked_id_idx',
                                            ])
    return df_new

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-s","--split",
    #                     help="The dataset split to use",
    #                     choices=['original','validation','validation_2','validation_3'],
    #                     type=str,
    #                     required=True)
    # args = parser.parse_args()
    # dataset_path = f"dataset/{args.split}"
    # isValidation = False
    # if args.split != 'original':
    #     isValidation = True
    splits = ['original','validation','validation_2','validation_3']
    for s in splits:
        dataset_path = f"dataset/{s}"
        if s != 'original':
            train = base_expanded_df(isValidation=True, save=True, path=dataset_path)
        else:
            test = base_expanded_df(isValidation=False, save=True, path=dataset_path)
