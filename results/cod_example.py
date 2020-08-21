from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np

from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
from models.cod import COD

if __name__ == "__main__":
    mat = loadmat(os.path.abspath(os.path.join(os.path.dirname(__file__), '..') + '/data/breastw.mat'))
    X = mat['X']
    y = mat['y'].ravel()

    # Generate sample data
    X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.4, random_state=42)

    clf = COD()
    clf.decision_function(X)
    a, b, c = clf.explain_outlier(69)
    print(a, b, c)
    a, b, c = clf.explain_outlier(97)
    print(a, b, c)