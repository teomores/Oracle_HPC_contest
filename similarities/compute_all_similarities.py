import os

splits = [#'original',
        #'validation',
        'validation_2',
        'validation_3']

similarities = [
                # 'ngrams_address.py',
                # 'ngrams_email.py',
                # 'ngrams_name.py',
                'ngrams_phone.py'
                ]
for sim in similarities:
    for sp in splits:
        print(f'Computing {sim} for split {sp}')
        os.system(f'python3 {sim} -s={sp}')
