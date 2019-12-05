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


def precision_at_k(resultTable: pd.DataFrame, trainingData: pd.DataFrame, testingData: pd.DataFrame) -> dict:
    """

    :param resultTable: columns: queried_record_id, predicted_record_id. Predicted_record_id is a list of the predicted record_id
                        (not predicted linked_id)
    :param trainingData: DO NOT set record_id as index
    :param testingData: DO NOT set record_id as index
    :return:
    """

    """
    Given a list of K predictions for each query, first retrieve the correct ID from the test data,
    then look in the training data the percentage of records that are actually relevant;

    For example, given query "1234-M", first retrieve the correct ID "1234" from the test data,
    then obtain from the training data all records that refer to "1234",
    and finally look how many of the records we have found are actually referring to "1234"
    """
    groupedTrainingRecords = trainingData.groupby("linked_id").apply(lambda x: list(x['record_id']))
    groupedTrainingRecords = groupedTrainingRecords.reset_index().rename(columns={0: 'record_id'})
    groupedTrainingRecords.linked_id = groupedTrainingRecords.linked_id.astype(str)
    resultTable = resultTable.sort_values(by='queried_record_id')
    testingData = testingData.sort_values(by='record_id')

    if resultTable.shape[0] != testingData.shape[0]:
        missing = set(testingData.record_id) - set(resultTable.queried_record_id)
        print(f'Missing some predictions: {missing}')
        return

    totalPrecision = 0.0
    numberOfPredictionsForRelevantRecords = 0

    allRecords = dict()

    resultTable['linked_id'] = testingData.linked_id.astype(str)
    resultTable = resultTable.merge(groupedTrainingRecords, how='left', left_on='linked_id', right_on='linked_id')
    #print(f"\ttime elapsed: {(time.time() - start):.2f} s")

    for (queriedRecordID, PredictedRecords, allRelevantRecords) in tqdm(
            zip(resultTable.queried_record_id, resultTable.predicted_record_id, resultTable.record_id)):

        try:
            selectedRelevantRecords = set(PredictedRecords) & set(allRelevantRecords)
        except:
            selectedRelevantRecords = set()
            allRelevantRecords = set()
        precision = 1
        if (len(allRelevantRecords) > 0):
            precision = len(selectedRelevantRecords) / len(PredictedRecords)
            numberOfPredictionsForRelevantRecords += len(PredictedRecords)

        totalPrecision += precision
        allRecords[queriedRecordID] = [queriedRecordID, precision, len(selectedRelevantRecords),
                                       len(allRelevantRecords)]

    # Store the results in a summary table;
    result_table = pd.DataFrame.from_dict(
        allRecords,
        orient='index',
        columns=["QueriedRecordID", "Precision@K", "SelectedRecords", "AllRelevantRecords"]
    )
    # Compute the filtered recall, which considers only queries with at least one relevant record in the training data;
    queries_with_relevant_records = result_table[result_table["AllRelevantRecords"] > 0]
    filtered_precision = np.mean(
        queries_with_relevant_records["SelectedRecords"] / numberOfPredictionsForRelevantRecords)

    return {
        "AveragePrecision": totalPrecision / resultTable.shape[0],
        "AverageFilteredPrecision": filtered_precision,
        "perQueryResult": result_table
    }


def recall_at_k(resultTable: pd.DataFrame, trainingData: pd.DataFrame, testingData: pd.DataFrame) -> dict:
    """

    :param resultTable: columns: queried_record_id, predicted_record_id. Predicted_record_id is a list of the predicted record_id
                        (not predicted linked_id)
    :param trainingData: DO NOT set record_id as index
    :param testingData: DO NOT set record_id as index
    :return:
    """
    """
    Given a list of K predictions for each query, first retrieve the correct ID from the test data,
    then look in the training data the percentage of records that have been successfully identified.

    For example, given query "1234-M", first retrieve the correct ID "1234" from the test data,
    then obtain from the training data all records that refer to "1234",
    and finally look how many of them we have found;
    """

    groupedTrainingRecords = trainingData.groupby("linked_id").apply(lambda x: list(x['record_id']))
    groupedTrainingRecords = groupedTrainingRecords.reset_index().rename(columns={0: 'record_id'})

    resultTable = resultTable.sort_values(by='queried_record_id')
    testingData = testingData.sort_values(by='record_id')

    if resultTable.shape[0] != testingData.shape[0]:
        missing = set(testingData.record_id) - set(resultTable.queried_record_id)
        print(f'Missing some predictions: {missing}')
        return

    totalRecall = 0.0

    allRecords = dict()

    resultTable['linked_id'] = testingData.linked_id.values
    resultTable = resultTable.merge(groupedTrainingRecords, how='left', left_on='linked_id', right_on='linked_id')
    # print(f"\ttime elapsed: {(time.time() - start):.2f} s")

    for (queriedRecordID, PredictedRecords, allRelevantRecords) in tqdm(
            zip(resultTable.queried_record_id, resultTable.predicted_record_id, resultTable.record_id)):

        try:
            selectedRelevantRecords = set(PredictedRecords) & set(allRelevantRecords)
        except:
            selectedRelevantRecords = set()
            allRelevantRecords = set()
        recall = 1
        if (len(allRelevantRecords) > 0):
            recall = len(selectedRelevantRecords) / len(allRelevantRecords)

        totalRecall += recall
        allRecords[queriedRecordID] = [queriedRecordID, recall, len(selectedRelevantRecords), len(allRelevantRecords)]

    # Store the results in a summary table;
    result_table = pd.DataFrame.from_dict(
        allRecords,
        orient='index',
        columns=["QueriedRecordID", "Recall@K", "SelectedRecords", "AllRelevantRecords"]
    )
    # Compute the filtered recall, which considers only queries with at least one relevant record in the training data;
    queries_with_relevant_records = result_table[result_table["AllRelevantRecords"] > 0]
    filtered_recall = np.mean(
        queries_with_relevant_records["SelectedRecords"] / queries_with_relevant_records["AllRelevantRecords"])

    return {
        "AverageRecall": totalRecall / resultTable.shape[0],
        "AverageFilteredRecall": filtered_recall,
        "perQueryResult": result_table
    }

def convert_phones(df_in):
    """
    This functions transforms the phone column from scientific notation to readable string
    format, e.g. 1.2933+E10 to 12933000000
    : param df_in : the original df with the phone in scientific notation
    : return : the clean df
    """
    df = df_in.copy()
    df.phone = df.phone.astype(str)
    df.phone = [p.split('.')[0] for p in df.phone]
    df.phone = df.phone.fillna('')
    return df

def threshold_matrix(mat: csr_matrix, thr: float = 0.9) -> csr_matrix:
    """
    This functions takes as input a sparse matrix and masks out all the elements
    that are below the set threshold.
    """
    mat.data[mat.data < thr] = 0
    return mat
