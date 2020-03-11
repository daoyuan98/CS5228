import numpy as np
import pandas as pd

def load_data(sett, n_fold=20, start_fold=0, missing_value='drop'):
    file_name = "../{}.csv".format(sett)
    data_whole = pd.read_csv(file_name, keep_default_na=True)
    data_whole = data_whole.replace(' ?', np.nan)
    
    if missing_value == 'drop':
        data_whole = data_whole.dropna(how='any')
    
    if sett == "train":
        train_index = []
        valid_index = []
        fold_index = 0
        one_fold = data_whole.shape[0] // n_fold
        
        for i in range(data_whole.shape[0]):
            if (i + 1) % one_fold == 0:
                fold_index += 1
            if fold_index == (start_fold % n_fold):
                valid_index.append(i)
            else:
                train_index.append(i)

        train_data, train_label = data_whole.iloc[train_index, :-1], data_whole.iloc[train_index ,-1:]
        valid_data, valid_label = data_whole.iloc[valid_index, :-1], data_whole.iloc[valid_index ,-1:]
        return train_data, train_label, valid_data, valid_label
    else:
        return data_whole