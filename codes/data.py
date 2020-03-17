import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

keys = {'age', 'workclass', 'fnlwgt', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'sex', 'capital-gain',
       'capital-loss', 'hours-per-week', 'native-country'}

non_continuous_keys = {'workclass', 'education', 'marital-status', 'occupation', 
                       'relationship', 'sex', 'native-country'}

non_continuous_key_values = {
     'workclass': [
        ' Federal-gov', ' Local-gov', ' Private', ' Self-emp-inc', ' Self-emp-not-inc', 
        ' State-gov', ' Without-pay', ' Never-worked'
        ],
     'education': [
        ' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th',
        ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate',
        ' HS-grad', ' Masters', ' Preschool', ' Prof-school',
        ' Some-college'
        ],
     'marital-status': [
        ' Divorced', ' Married-AF-spouse', ' Married-civ-spouse',
        ' Married-spouse-absent', ' Never-married', ' Separated',
        ' Widowed'
        ],
     'occupation': [' Adm-clerical', ' Armed-Forces', ' Craft-repair',
       ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners',
       ' Machine-op-inspct', ' Other-service', ' Priv-house-serv',
       ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support',
       ' Transport-moving'
       ],
     'relationship': [' Husband', ' Not-in-family', ' Other-relative', ' Own-child',
       ' Unmarried', ' Wife'
       ],
     'sex': [
        ' Female', ' Male'
       ], 
     'native-country': [
        ' Cambodia', ' Canada', ' China', ' Columbia', ' Cuba',
        ' Dominican-Republic', ' Ecuador', ' El-Salvador', ' England',
        ' France', ' Germany', ' Greece', ' Guatemala', ' Haiti',
        ' Holand-Netherlands', ' Honduras', ' Hong', ' Hungary', ' India',
        ' Iran', ' Ireland', ' Italy', ' Jamaica', ' Japan', ' Laos',
        ' Mexico', ' Nicaragua', ' Outlying-US(Guam-USVI-etc)', ' Peru',
        ' Philippines', ' Poland', ' Portugal', ' Puerto-Rico',
        ' Scotland', ' South', ' Taiwan', ' Thailand', ' Trinadad&Tobago',
        ' United-States', ' Vietnam', ' Yugoslavia'
      ]
}

def one_hot(data, sett):
    continuous_keys = keys - non_continuous_keys
    res = pd.DataFrame(data[continuous_keys])
    # print(res)
    if sett == "train":
        label = pd.DataFrame(data['exceeds50K'])

    one_hot_components = []
    for key in non_continuous_keys:
        one_hot_encoder = OneHotEncoder()
        label_encoder = LabelEncoder()

        label_encoder_labels = label_encoder.fit_transform(data[key])
        feature_arr = one_hot_encoder.fit_transform(data[[key]]).toarray()

        new_labels = list(label_encoder.classes_)
        features = pd.DataFrame(feature_arr, columns=new_labels)
        one_hot_components.append(features) 
    if sett == "train":
        ret = pd.concat([res, *one_hot_components, label], axis=1)
    else:
        ret = pd.concat([res, *one_hot_components], axis=1)
    return ret

def load_data(sett, n_fold=20, start_fold=19, missing_value='drop', do_one_hot=True, numpy=True):
    file_name = "../{}.csv".format(sett)
    data_whole = pd.read_csv(file_name, keep_default_na=True)
    data_whole = data_whole.replace(' ?', np.nan)

    if missing_value == 'drop':
        print("drop missing value")
        data_na_dropped = data_whole.dropna(how='any')
        data = data_na_dropped.reset_index()
    
    if missing_value == 'fill':
        print("fill missing value")
        data = data_whole.fillna(method='ffill')
        data = data.fillna(method='bfill')

    if missing_value == 'mean':
        print('fill missing value with mean')
        data = data_whole.fillna(data_whole.mean())
        
        data = data_whole.fillna(method='ffill')
        data = data.fillna(method='bfill')

    if do_one_hot:
        data_whole =  pd.get_dummies(data)

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

        if missing_value == "fill" and ' Never-worked' in data_whole.keys():
            data_whole = data_whole.drop(columns=[' Never-worked'])

        # train_data, train_label = data_whole.iloc[train_index, :-1], data_whole.iloc[train_index ,-1:]
        # valid_data, valid_label = data_whole.iloc[valid_index, :-1], data_whole.iloc[valid_index ,-1:]

        data = data_whole.drop(columns='exceeds50K')
        label = data_whole['exceeds50K']

        train_data = data.iloc[train_index, :]
        train_label = label.iloc[train_index]

        valid_data = data.iloc[valid_index, :]
        valid_label = label.iloc[valid_index]

        if numpy:
            return train_data.to_numpy(), train_label.to_numpy(), valid_data.to_numpy(), valid_label.to_numpy()
        else:
            return train_data, train_label, valid_data, valid_label
    else:
        if 'native-country_ Holand-Netherlands' in data_whole.keys():
            print("drop ", 'native-country_ Holand-Netherlands')
            data_whole = data_whole.drop(columns=['native-country_ Holand-Netherlands'])
        # if 'workclass_ Never-worked' in data_whole.keys():
        #     print('drop workclass')
        #     data_whole = data_whole.drop(columns=['workclass_ Never-worked'])
        # print(data_whole.head())
        if numpy:
            return data_whole.to_numpy()
        else:
            return data_whole


if __name__ == '__main__':
    train_data, train_label, valid_data, valid_label = load_data("train", missing_value='fill', numpy=False)
    test_data = load_data("test", missing_value='fill', numpy=False)


    train_countries = set(np.unique(train_data.keys()))
    test_countries = set(np.unique(test_data.keys()))
    print(train_countries)
    print(test_countries)
    print("diff: ", test_countries - train_countries)
    print("diff: ", train_countries - test_countries)
    # diff:  {' Holand-Netherlands'}
    print(sum([len(val) for key, val in non_continuous_key_values.items()]))