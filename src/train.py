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
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import roc_auc_score
import json
import yaml

params = yaml.safe_load(open("params.yaml"))["train"]
params2 = yaml.safe_load(open("params.yaml"))["featurize"]



if len(sys.argv) != 4:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-file\n")
    sys.exit(1)

input = sys.argv[1]
output_model = sys.argv[2]
output_score = sys.argv[3]

split = params["split"]
seed = 123  # random.seed(params["seed"])
n_est = params["n_est"]
max_number = params2["max_number"]

df_input = pd.read_csv(input)
labels = pd.read_csv("./data/initial_data.csv")["class"][:max_number]

X_train, X_test, y_train, y_test = train_test_split(df_input, #Filtered Descriptors 
                                                    labels,  #Vector containing each compound's class
                                                    test_size= 0.1, 
                                                    shuffle= True, 
                                                    random_state=123, 
                                                    stratify= labels
                                                   )

print(f'Blockers - training set: {pd.value_counts(y_train)[1]} compounds')
print(f'Non-blockers - training set: {pd.value_counts(y_train)[0]} compounds')
print(f'Total compounds - training set: {len(y_train)}')
print(f'Blockers - test set: {pd.value_counts(y_test)[1]} compounds')
print(f'Non-blockers - test set: {pd.value_counts(y_test)[0]} compounds')
print(f'Total compounds - test set: {(len(y_test))}')

clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=123)

rf_pipeline = Pipeline([("st_scaler", StandardScaler()),
                        ("rf_clf", RandomForestClassifier(n_estimators=100, max_depth=4, random_state=123))])


os.makedirs(os.path.join("models"), exist_ok=True)

os.makedirs(os.path.join("metrics"), exist_ok=True)

with open(output_model, "wb") as fd:
    pickle.dump(clf, fd)


y_predict_rf = cross_val_predict(rf_pipeline,  # pipeline containing the transformer and the estimator.
                              X_train,  # Features
                              y_train,  # Labels
                              cv=10,
                              method='predict',
                              n_jobs=-1  # Use all available cores.
                              )

print(cm(y_train, y_predict_rf), '\n') #Confusion Matrix

print(clf_rep(y_train, y_predict_rf)) #Classification report


y_probas_rf = cross_val_predict(rf_pipeline, #pipeline containing the transformer and the estimator.
                                 X_train, #Features
                                 y_train, #Labels
                                 cv=10,   
                                 method='predict_proba', #Returns a probability that a given instance belongs to a specific class.
                                 n_jobs=-1 #Use all available cores.
                             )

y_scores_rf = y_probas_rf[:,1] #score = proba of positive class.

print(clf_rep)
score_dict = {
    "roc-auc": round(roc_auc_score(y_train, y_scores_rf), 2)
}


with open(f"{output_score}.json", "w") as outfile:
    json.dump(score_dict, outfile)

print('AUC score for RF model: ', score_dict['roc-auc'],'\n')