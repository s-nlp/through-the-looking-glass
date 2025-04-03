import argparse
import os
import torch
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from utils.datasets import get_dataset


def fit_predict(train_x, train_y, test_x, i):
    scaler = StandardScaler()
    logistic = LogisticRegression(max_iter=100, tol=0.1)
    pipe = Pipeline(steps=[("scaler", scaler), ("logistic", logistic)])

    pipe.fit(train_x[:, i], train_y)
    pred = pipe.predict(test_x[:, i])

    return pred


def train_fold(train_x, train_y, val_x, val_y, val_acc, n):
    for i in range(n):
        pred = fit_predict(train_x, train_y, val_x, i)
        val_acc[i].append(accuracy_score(val_y, pred))


def test_fold(train_x, train_y, test_x, test_y, max_i, test_acc):
    pred = fit_predict(train_x, train_y, test_x, max_i)
    test_acc.append(accuracy_score(test_y, pred))


def get_fair_results(file, labels, SEED=12):
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    label = np.array(labels)

    pd_train = torch.load(file).type(torch.float32)
    features = pd_train.transpose(1, 0)

    n = pd_train.shape[0]
    val_acc = {i: [] for i in range(n)}

    for i, (train_index, test_index) in enumerate(kf.split(features)):
        train_index, val_index = train_test_split(train_index, test_size=0.25, random_state=SEED)

        train_x = features[train_index]
        train_y = label[train_index]

        val_x = features[val_index]
        val_y = label[val_index]

        test_x = features[test_index]
        test_y = label[test_index]

        train_fold(train_x, train_y, val_x, val_y, val_acc, n)

    max_acc = 0
    max_i = 0

    accs = []
    for i in range(1, n):
        new_acc = np.mean(val_acc[i])
        accs.append(new_acc)
        if new_acc > max_acc:
            max_acc = new_acc
            max_i = i

    test_acc = []
    for i, (train_index, test_index) in enumerate(kf.split(features)):
        train_x = features[train_index]
        train_y = label[train_index]

        test_x = features[test_index]
        test_y = label[test_index]

        test_fold(train_x, train_y, test_x, test_y, max_i, test_acc)

    return np.mean(test_acc), np.std(test_acc), val_acc


def evaluate_linear_probing(args):
    path = args.hidden_states_path
    dataset = path.split("/")[-1].split("_")[0]

    _, labels = get_dataset(dataset)
    labels = labels
    mu, *_ = get_fair_results(path, labels)
    print(f"Cross-validation accuracy is: {mu}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate linear probing on hidden states.")
    parser.add_argument("--hidden_states_path", required=True, type=str, help="Name of the model.")
    
    args = parser.parse_args()
    evaluate_linear_probing(args)
