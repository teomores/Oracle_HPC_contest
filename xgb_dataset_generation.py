import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy import *
from scipy.sparse import *
from pathlib import Path

from features.compute_editdistance import compute_editdistance
from features.target import target


def base_expanded_df(alpha = 0.2, beta = 0.05, k = 20, isValidation=False, save=False):
    if isValidation:
        sim_name = load_npz('jaccard_tfidf_name_validation.npz')
        sim_email = load_npz('jaccard_tfidf_email_validation.npz')
        sim_phone = load_npz('jaccard_tfidf_phone_validation.npz')
        df_train = pd.read_csv('dataset/validation/train.csv', escapechar="\\")
        df_test = pd.read_csv('dataset/validation/test.csv', escapechar="\\")
    else:
        sim_name = load_npz('jaccard_tfidf_name_original.npz')
        sim_email = load_npz('jaccard_tfidf_email_original.npz')
        sim_phone = load_npz('jaccard_tfidf_phone_original.npz')
        df_train = pd.read_csv('dataset/original/train.csv', escapechar="\\")
        df_test = pd.read_csv('dataset/original/test.csv', escapechar="\\")

    hybrid = sim_name + alpha * sim_email + beta * sim_phone

    df_train = df_train.sort_values(by=['record_id']).reset_index(drop=True)
    df_test = df_test.sort_values(by=['record_id']).reset_index(drop=True)

    """
    linid_ = []
    linid_idx = []
    linid_score = []
    linid_name_cosine = []
    linid_email_cosine = []
    linid_phone_cosine = []
    linid_record_id = []

    tr = df_train[['record_id', 'linked_id']]
    for x in tqdm(range(df_test.shape[0])):
        df = df_train.loc[hybrid[x].nonzero()[1][hybrid[x].data.argsort()[::-1]],:].drop_duplicates(['linked_id'])[:k]
        indices = hybrid[x].nonzero()[1][hybrid[x].data.argsort()[::-1]]
        df = tr.loc[indices, :][:k]
        linid_.append(df['linked_id'].values)
        linid_idx.append(df.index)
        linid_record_id.append(df.record_id.values)
        linid_score.append(np.sort(hybrid[x].data)[::-1][:k])
        linid_name_cosine.append(np.sort(sim_name[x].data)[::-1][:k])
        linid_email_cosine.append(np.sort(sim_email[x].data)[::-1][:k])
        linid_phone_cosine.append(np.sort(sim_phone[x].data)[::-1][:k])

    """

    linid_score = []
    linid_name_cosine = []
    linid_email_cosine = []
    linid_phone_cosine = []
    linid_record_id = []
    k = 10
    indices = []
    for x in tqdm(range(df_test.shape[0])):
        indices.append(hybrid[x].nonzero()[1][hybrid[x].data.argsort()[::-1]])

    linked_id_list = []
    relevant_idx = []
    num_diff_lin_id = 30
    # use indices wrt to loc, much more faster
    # avoid drop_duplicates, simply check whether the linked_id is already in the list
    dict_index_linked_id = dict(zip(df_train.index, df_train.linked_id))
    print("Retrieving linked ids from df_train...")
    for x in tqdm(indices):
        tmp = []
        idx = []
        for l in x:
            if len(tmp) < num_diff_lin_id:
                ind = dict_index_linked_id[l]
                if ind not in tmp:
                    tmp.append(ind)
                    idx.append(l)
            else:
                continue
        linked_id_list.append(tmp)
        relevant_idx.append(idx)
    for x in tqdm(range(df_test.shape[0])):
        linid_score.append([hybrid[x, t] for t in relevant_idx[x]])
        linid_name_cosine.append([sim_name[x, t] for t in relevant_idx[x]])
        linid_email_cosine.append([sim_email[x, t] for t in relevant_idx[x]])
        linid_phone_cosine.append([sim_phone[x, t] for t in relevant_idx[x]])

    df = pd.DataFrame()
    df['queried_record_id'] = df_test.record_id
    df['predicted_record_id'] = linked_id_list
    #df['predicted_record_id'] = linid_
    #df['predicted_record_id_record'] = linid_record_id
    df['cosine_score'] = linid_score
    df['name_cosine'] = linid_name_cosine
    df['email_cosine'] = linid_email_cosine
    df['phone_cosine'] = linid_phone_cosine
    #df['linked_id_idx'] = linid_idx
    df['linked_id_idx'] = relevant_idx


    df_new = expand_df(df)

    if save:
        if isValidation:
            df_new.to_csv("dataset/expanded/base_expanded_train.csv", index=False)
        else:
            df_new.to_csv("dataset/expanded/base_expanded_test.csv", index=False)

    return df_new



def expand_df(df):
    df_list = []
    for (q, pred, score, s_name, s_email, s_phone, idx) in tqdm(
            zip(df.queried_record_id, df.predicted_record_id, df.cosine_score,
                df.name_cosine, df.email_cosine, df.phone_cosine, df.linked_id_idx)):
        for x in range(len(pred)):
            df_list.append((q, pred[x], score[x], s_name[x], s_email[x], s_phone[x], idx[x]))

    df_new = pd.DataFrame(df_list, columns=['queried_record_id', 'predicted_record_id', 'cosine_score', 'name_cosine',
                                            'email_cosine', 'phone_cosine', 'linked_id_idx',
                                            ])
    return df_new


def adding_features(df, isValidation=True):
    """

    :param df: expanded dataset. Call it after execute base_expanded_df
    :param isValidation:
    :return:
    """

    curr_dir = Path(__file__).absolute().parent
    if isValidation:
        feat_dir = curr_dir.joinpath("dataset/validation/feature/")
    else:
        feat_dir = curr_dir.joinpath("dataset/original/feature/")

    if isValidation:
        df = target(df)

    email_pop = pd.read_csv( feat_dir.joinpath("email_popularity.csv"))
    linked_id_pop = pd.read_csv( feat_dir.joinpath("linked_id_popularity.csv"))
    name_pop = pd.read_csv( feat_dir.joinpath("name_popularity.csv"))
    nonnull_addr = pd.read_csv( feat_dir.joinpath("number_of_non_null_address.csv"))
    nonnull_email = pd.read_csv( feat_dir.joinpath("number_of_non_null_email.csv"))
    nonnull_phone = pd.read_csv( feat_dir.joinpath("number_of_non_null_phone.csv"))
    phone_pop = pd.read_csv( feat_dir.joinpath("phone_popularity.csv"))
    name_length = pd.read_csv( feat_dir.joinpath("test_name_length.csv"))
    print(df.columns)
    df['editdistance'] = compute_editdistance(df, validation=isValidation)
    df = df.merge(email_pop, how='left', left_on='queried_record_id', right_on='record_id').drop('record_id', axis=1)
    print(df.columns)
    df = df.merge(linked_id_pop, how='left', left_on='predicted_record_id', right_on='linked_id').drop('linked_id', axis=1).rename(
        columns={'popularity': 'linked_id_popularity'})
    print(df.columns)
    df = df.merge(name_pop, how='left', left_on='queried_record_id', right_on='record_id').drop('record_id', axis=1)
    print(df.columns)
    df = df.merge(nonnull_addr, how='left', left_on='predicted_record_id', right_on='linked_id')
    print(df.columns)
    df = df.drop('linked_id', axis=1)
    df = df.merge(nonnull_email, how='left', left_on='predicted_record_id', right_on='linked_id').drop('linked_id',
                                                                                                           axis=1)
    df = df.merge(nonnull_phone, how='left', left_on='predicted_record_id', right_on='linked_id').drop('linked_id',
                                                                                                           axis=1)
    df = df.merge(phone_pop, how='left', left_on='queried_record_id', right_on='record_id').drop('record_id',
                                                                                                     axis=1)
    df = df.merge(name_length, how='left', left_on='queried_record_id', right_on='record_id').drop('record_id',
                                                                                                       axis=1)
    df = df.fillna(0)

    df['linked_id_popularity'] = df.linked_id_popularity.astype(int)
    df['null_address'] = df.null_address.astype(int)
    df['null_email'] = df.null_email.astype(int)
    df['null_phone'] = df.null_phone.astype(int)

    return df
