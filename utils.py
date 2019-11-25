from tqdm import tqdm
from scipy import *
from scipy.sparse import *
import pandas as pd

def get_sub(sim, df_train, df_test, sub_name='mimmo'):
    """
    This function generates a submission-style pandas dataframe from the similarity
    and writes the dataframe to a csv file named as the sub_name parameter
    : param sim : similarity in CSR format
    : param df_train : the train pandas dataframe
    : param df_test : the test pandas dataframe
    : param sub_name : the name of the file of the submission
    : return : the pandas dataframe
    """
    # first make sure df_train and df_test are sorted by record_id
    print("Sorting dataframes...")
    df_train = df_train.sort_values(by=['record_id']).reset_index(drop=True)
    df_test = df_test.sort_values(by=['record_id']).reset_index(drop=True)
    # then extract top indices sorting
    print("Sorting similarity to get top indices...")
    indices = []
    for x in tqdm(range(sim.shape[0])):
        if x == 0:
            indices.append(sim[x].nonzero()[1][sim[x].data[1:].argsort()[::-1]])
        else:
            indices.append(sim[x].nonzero()[1][sim[x].data.argsort()[::-1]])
    linked_id_list = []
    num_diff_lin_id = 10
    # use indices wrt to loc, much more faster
    # avoid drop_duplicates, simply check whether the linked_id is already in the list
    dict_index_linked_id =dict(zip(df_train.index, df_train.linked_id))
    print("Retrieving linked ids from df_train...")
    for x in tqdm(indices):
        tmp = []
        for l in x:
            if len(tmp)<num_diff_lin_id:
                ind = dict_index_linked_id[l]
                if ind not in tmp:
                    tmp.append(ind)
            else:
                continue
        linked_id_list.append(tmp)
    # the create sub
    print("Creating the sub...")
    sub = pd.DataFrame()
    sub['queried_record_id'] = df_test.record_id
    sub['predicted_record_id'] = linked_id_list
    print('Exploding list to string...')
    strings = []
    for t in tqdm(sub.predicted_record_id):
        strings.append(' '.join([str(x) for x in t]))
    sub['predicted_record_id'] = strings
    print(f"Writing to {sub_name}.csv...")
    sub.to_csv(f'{sub_name}.csv', index=False)
    print('DONE!')
    return sub
