import os

# first of all: create the validation
os.system('python split_train_validation.py')
# now create all the similarities
os.system('python similarities/compute_all_similarities.py')
# compute features
os.system('python features/case_typo.py')
os.system('python features/linked_id_popularity.py')
os.system('python features/name_popularity.py')
os.system('python features/email_popularity.py')
os.system('python features/number_of_non_null_address.py')
os.system('python features/number_of_non_null_email.py')
os.system('python features/number_of_non_null_phone.py')
os.system('python features/phone_popularity.py')
os.system('python features/test_name_length.py')
# now create the dataframe for LightGBM
os.system('python create_expanded_dataset.py')
# finally run LightGBM
os.system('python LightGBM_full.py')
