import os
import sys
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report as clf_rep
from sklearn.model_selection import train_test_split
import yaml

params = yaml.safe_load(open("params.yaml"))["train"]
params2 = yaml.safe_load(open("params.yaml"))["featurize"]

max_number = params2["max_number"]

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-file\n")
    sys.exit(1)

input = sys.argv[1]
output_model = sys.argv[2]

split = params["split"]
seed = 123  # random.seed(params["seed"])
n_est = params["n_est"]
method = params["method"]


def get_pipeline(seed, n_est, clf_model):
    """
    Data scaling and estimation within a pipeline
    """

    rf_pipeline = Pipeline([("std_scaler", StandardScaler()),
                            ("rf_clf", clf_model)])

    return rf_pipeline


def get_metrics(pipeline, prediction):
    """
    Function to get metrics from pipeline and predicition
    """
    return cm(pipeline, prediction)


def get_report(train, prediction):
    print(clf_rep(train, prediction))


df = pd.read_csv(input)
labels = pd.read_csv("./data/initial_data.csv")["label"][:max_number]
X_train, X_test, y_train, y_test = train_test_split(df,  # Filtered Descriptors
                                                    labels,  # Vector containing each compound's class
                                                    test_size=split,
                                                    shuffle=True,
                                                    random_state=seed,
                                                    stratify=labels
                                                    )
clf = RandomForestClassifier(
    n_estimators=n_est, max_depth=4, random_state=seed)

pipe = get_pipeline(seed, n_est, clf)

clf.fit(X_train, y_train)


os.makedirs(os.path.join("models"), exist_ok=True)


with open(output_model, "wb") as fd:
    pickle.dump(clf, fd)
