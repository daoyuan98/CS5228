import numpy as np

from data import *
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score

def assemble_predict(models, datas, weights=None, threshold=0.5):

    if weights is None:
        weights = [1.] * len(models)
    pred = np.zeros((datas[0].shape[0], ))

    for i, model in enumerate(models):
        pred = pred + model.predict(datas[i])

    pred = pred / sum(weights)
    res = np.zeros((datas[0].shape[0], ))
    res[pred >= threshold] = 1

    return res


def get_acc(pred, gt):
    pred = pred.reshape((pred.shape[0], ))
    gt = gt.reshape((gt.shape[0], ))
    return accuracy_score(gt, pred)

def get_f1_score(pred, gt):
    pred = pred.reshape((pred.shape[0], ))
    gt = gt.reshape((gt.shape[0], ))
    return accuracy_score(gt, pred)

def get_score(clf, data, label, metrics='f1'):
    pred = clf.predict(data)
    if metrics == 'f1':
        return get_f1_score(pred, label)
    if metrics == 'acc':
        return get_f1_score(pred, label)

def get_name(clf):
    raw = str(clf)
    return raw[:raw.index('(')]

def output(pred, file):
    file_path = '../output/{}.csv'.format(file)
    with open(file_path, 'w+') as f:
        f.write('{},{}\n'.format("id", "prediction"))
        for i in range(len(pred)):
            f.write('{},{}\n'.format(i+1, int(pred[i])))

def scale(split1, split2, split3):
    whole = np.vstack([split1, split2, split3])
    # print(whole.shape)
    for i in range(6):
        max_ = np.max(whole[:, i])
        min_ = np.min(whole[:, i])
        whole[:, i] = (whole[:, i] - min_) / (max_ - min_)
    return whole[:split1.shape[0], :], whole[split1.shape[0]:split2.shape[0]+split1.shape[0], :], whole[-split3.shape[0]:, :]

def main():
    
    print('--'*50)

    train_data, train_label, valid_data, valid_label = load_data("train", n_fold=30,  missing_value='mean', do_one_hot=True)
    test_data = load_data("test", missing_value='mean', do_one_hot=True)

    # train_scaled, test_scaled, valid_scaled = scale(train_data, test_data, valid_data)
    # exit(0)
    # print(train_scaled.shape, test_scaled.shape, valid_scaled.shape)

    train_label = train_label.reshape((train_label.shape[0], ))
    valid_label = valid_label.reshape((valid_label.shape[0], ))


    # np.random.seed(19)
    random_seeds = [9, 19, 29, 199, 299, 1999, 2999, 1919]
    print(train_data.shape)
    models = []

    def eval_model(clf, train_data, valid_data, add=True, metric='f1'):
        acc_on_train = get_score(clf, train_data, train_label, metric)
        acc_on_valid = get_score(clf, valid_data, valid_label, metric)

        print("[{}] {} on train: {:.4f}".format(get_name(clf), metric, acc_on_train))
        print("[{}] {} on valid: {:.4f}".format(get_name(clf), metric, acc_on_valid))

        print('--'*50)

        if add:
            models.append(clf)
        return acc_on_valid
    
    def lr_model():
        from sklearn.linear_model import LogisticRegressionCV
        clf = LogisticRegressionCV(Cs=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4], cv=20, random_state=np.random.randint(1, 10000), max_iter=1000).fit(train_data, train_label)
        eval_model(clf, train_data, valid_data)

    def mlp_model():
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(15), alpha=0.01, random_state=np.random.randint(1, 10000), max_iter=10000).fit(train_scaled, train_label)
        eval_model(clf, train_scaled, valid_scaled)

    def svm_model():
        from sklearn.svm import SVC
        clf = SVC().fit(train_data, train_label)
        eval_model(clf)

    def random_forest_model(depth=14):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=depth, n_estimators=50, random_state=random_seeds.pop()).fit(train_data, train_label)
        eval_model(clf, train_data, valid_data)

    def adaboost_classifier():
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier(n_estimators=50, random_state=random_seeds.pop()).fit(train_data, train_label)
        eval_model(clf)

    def xgboost_classifier(n_est=220, depth=3):
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=n_est, max_depth=depth, random_state=random_seeds.pop()).fit(train_data, train_label)
        eval_model(clf, train_data, valid_data)


    print('--'*50)

    # define models
    # lr_model()
    # mlp_model()
    # mlp_model()
    # mlp_model()
    # svm_model()

    # random_forest_model(12)
    # random_forest_model(11)
    # random_forest_model(12)
    # random_forest_model(13)
    # random_forest_model(14)

    # adaboost_classifier()

    xgboost_classifier(220, 3)
    xgboost_classifier(210, 3)
    xgboost_classifier(230, 3)
    # xgboost_classifier(100, 4)
    # xgboost_classifier(1000, 3)
    
    # final model
    # random_forest_model(14)
    # xgboost_classifier()

    valid_pred = assemble_predict(models, [valid_data]*len(models))
    print("assemble acc on valid: ", get_acc(valid_pred, valid_label))

    test_pred = assemble_predict(models, [test_data]*len(models))

    import datetime
    output(test_pred, str(datetime.datetime.now()).replace("-", "").replace(".", "").replace(" ", "").replace(":", ""))

if __name__ == '__main__':
    main()

