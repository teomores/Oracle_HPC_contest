import os

# first of all: create the validation
# os.system('python split_train_validation.py')
# # now create all the similarities
# mydir_new = os.chdir('similarities')
# os.system('python similarities/compute_all_similarities.py')
# # # compute features
# mydir_new = os.chdir('../features')
# os.system('python case_typo.py')
# os.system('python linked_id_popularity.py')
# os.system('python name_popularity.py')
# os.system('python email_popularity.py')
# os.system('python number_of_non_null_address.py')
# os.system('python number_of_non_null_email.py')
# os.system('python number_of_non_null_phone.py')
# os.system('python phone_popularity.py')
# os.system('python test_name_length.py')
# # now create the dataframe for LightGBM
# mydir_new = os.chdir('..')
# os.system('python create_expanded_dataset.py')
# # finally run LightGBM
# os.system('python LightGBM_full.py')
os.system('sub_evaluation.py')
