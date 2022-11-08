import json
import math
import os
import pickle
import sys

import pandas as pd
from sklearn import metrics
from sklearn import tree
from dvclive import Live
from matplotlib import pyplot as plt

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

def predict(pipeline, xtrain, ytrain, method: str, cross_validation: int = 10, ):

    y_predict = cross_val_predict(pipeline,  # pipeline containing the transformer and the estimator.
                                  xtrain,  # Features
                                  ytrain,  # Labels
                                  cv=cross_validation,
                                  method=method,
                                  n_jobs=-1  # Use all available cores.
                                  )

    return y_predict