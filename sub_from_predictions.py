import pandas as pd
import numpy as np
import os

sub = pd.read_csv("lgb_predictions_3ngramsname_2glialtri.csv")
sub.ordered_linked = [eval(x) for x in sub.ordered_linked]
sub['predicted_record_id'] = [" ".join([str(y) for y in x[:10]]) for x in sub.ordered_linked]

if os.path.exists('daje.csv'):
    os.remove('daje.csv')
sub[['queried_record_id', 'predicted_record_id']].to_csv("daje.csv", index=False)
