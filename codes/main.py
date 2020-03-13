import numpy as np

from data import *
from sklearn.linear_model import LogisticRegressionCV

def assemble_predict(models, data, weights=None, threshold=0.5):

    print(data.shape)

    if weights is None:
        weights = [1.] * len(models)
    pred = np.zeros((data.shape[0], ))

    for model in models:
        pred = pred + model.predict(data)

    pred = pred / sum(weights)
    res = np.zeros((data.shape[0], ))
    res[pred >= threshold] = 1

    return res


def get_acc(pred, gt):
    pred = pred.reshape((pred.shape[0], ))
    gt = gt.reshape((gt.shape[0], ))
    return np.sum(pred == gt) / pred.shape[0]


def output(pred, file):
    file_path = '../output/{}.csv'.format(file)
    with open(file_path, 'w+') as f:
        f.write('{},{}\n'.format("id", "prediction"))
        for i in range(len(pred)):
            f.write('{},{}\n'.format(i+1, int(pred[i])))

def main():
    
    print('--'*50)

    train_data, train_label, valid_data, valid_label = load_data("train", missing_value='drop')
    test_data = load_data("test", missing_value='fill')

    train_label = train_label.reshape((train_label.shape[0], ))
    valid_label = valid_label.reshape((valid_label.shape[0], ))

    np.random.seed(19)
    print(train_data.shape)
    models = []
    
    def lr_model():
        from sklearn.linear_model import LogisticRegressionCV
        clf = LogisticRegressionCV(cv=10, random_state=np.random.randint(1, 10000), max_iter=1000).fit(train_data, train_label)
        acc_on_train = clf.score(train_data, train_label)
        acc_on_valid = clf.score(valid_data, valid_label)

        print("[logistic regression]acc on train: ", acc_on_train)
        print("[logistic regression]acc on valid: ", acc_on_valid)
        print('--'*50)

        models.append(clf)

    def mlp_model():
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(random_state=np.random.randint(1, 10000), max_iter=10000).fit(train_data, train_label)
        acc_on_train = clf.score(train_data, train_label)
        acc_on_valid = clf.score(valid_data, valid_label)

        print("[multi-layer perceptron]acc on train: ", acc_on_train)
        print("[multi-layer perceptron]acc on valid: ", acc_on_valid)
        print('--'*50)
        if acc_on_valid > 75:
            models.append(clf)

    def svm_model():
        from sklearn.svm import SVC
        clf = SVC().fit(train_data, train_label)
        
        acc_on_train = clf.score(train_data, train_label)
        acc_on_valid = clf.score(valid_data, valid_label)

        print("[svm]acc on train: ", acc_on_train)
        print("[svm]acc on valid: ", acc_on_valid)
        print('--'*50)

        models.append(clf)

    def random_forest_model(depth=14):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=depth, random_state=np.random.randint(1, 10000)).fit(train_data, train_label)

        acc_on_train = clf.score(train_data, train_label)
        acc_on_valid = clf.score(valid_data, valid_label)

        print("[random forest]acc on train: ", acc_on_train)
        print("[random forest]acc on valid: ", acc_on_valid)
        print('--'*50)

        models.append(clf)
    
    print('--'*50)

    # define models
    # lr_model()
    # mlp_model()
    # mlp_model()
    # mlp_model()
    # svm_model()

    random_forest_model(12)
    random_forest_model(12)
    random_forest_model(12)
    random_forest_model(12)

    valid_res = assemble_predict(models, valid_data)
    print("assemble acc on valid: ", get_acc(valid_res, valid_label))

    test_pred = assemble_predict(models, test_data)

    import datetime
    output(test_pred, str(datetime.datetime.now()).replace("-", "").replace(".", "").replace(" ", "").replace(":", ""))

if __name__ == '__main__':
    main()

