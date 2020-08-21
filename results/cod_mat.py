from __future__ import division
from __future__ import print_function

import os
import sys
from time import time
import pdb
# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lmdd import LMDD
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sod import SOD
from pyod.models.sos import SOS

from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
from models.cod_exp import COD

# Define data file and read X and y
mat_file_list = [
                 'arrhythmia.mat',
                 'breastw.mat',
                 'cardio.mat',
                 'cover.mat',
                 'ionosphere.mat',
                 'lympho.mat',
                 'mammography.mat',
                 'optdigits.mat',
                 'pima.mat',
                 'satellite.mat',
                 'satimage-2.mat',
                 'speech.mat',
                 'wbc.mat',
                 'wine.mat']

# define the number of iterations
n_ite = 10
n_classifiers = 6

df_columns = ['Data', '# Samples', '# Dimensions', 'Outlier Perc',
              'COD_L', 'COD_R', 'COD_B', 'COD_S', 'COD_M', 'COD']

# initialize the container for saving the results
roc_df = pd.DataFrame(columns=df_columns)
prn_df = pd.DataFrame(columns=df_columns)
ap_df = pd.DataFrame(columns=df_columns)
time_df = pd.DataFrame(columns=df_columns)

for j in range(len(mat_file_list)):
    mat_file = mat_file_list[j]
    mat = loadmat(os.path.abspath(os.path.join(os.path.dirname(__file__), '..') + '/data/' + mat_file))
    X = mat['X']
    y = mat['y'].ravel()
    if X.shape[0] > 10000:
      index = np.random.choice(X.shape[0], 10000, replace=False)  
      X = X[index]
      y = y[index]

    outliers_fraction = np.count_nonzero(y) / len(y)
    outliers_percentage = round(outliers_fraction * 100, ndigits=4)

    # construct containers for saving results
    roc_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    prn_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    ap_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    time_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]

    roc_mat = np.zeros([n_ite, n_classifiers])
    prn_mat = np.zeros([n_ite, n_classifiers])
    ap_mat = np.zeros([n_ite, n_classifiers])
    time_mat = np.zeros([n_ite, n_classifiers])

    for i in range(n_ite):
        print("\n... Processing", mat_file, '...', 'Iteration', i + 1)
        random_state = np.random.RandomState(i)

        # 60% data for training and 40% for testing
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.4, random_state=random_state)

        # standardizing data for processing
        X_train_norm, X_test_norm = standardizer(X_train, X_test)

        classifiers = {'COD_L': COD(contamination=outliers_fraction, tail='left'),
                       'COD_R': COD(contamination=outliers_fraction, tail='right'),
                       'COD_B': COD(contamination=outliers_fraction, tail='both'),
                       'COD_S': COD(contamination=outliers_fraction, tail='skew'),
                       'COD_M': COD(contamination=outliers_fraction, tail='max'),
                       'COD': COD(contamination=outliers_fraction)
                       }
        classifiers_indices = {
            'COD_L': 0,
            'COD_R': 1,
            'COD_B': 2,
            'COD_S': 3,
            'COD_M': 4,
            'COD': 5
        }

        for clf_name, clf in classifiers.items():
            t0 = time()
            clf.fit(X_train_norm)
            test_scores = clf.decision_function(X_test_norm)
            t1 = time()
            duration = round(t1 - t0, ndigits=4)
            test_scores = np.nan_to_num(test_scores)

            roc = round(roc_auc_score(y_test, test_scores), ndigits=4)
            prn = round(precision_n_scores(y_test, test_scores), ndigits=4)
            ap = round(average_precision_score(y_test, test_scores), ndigits=4)

            print('{clf_name} ROC:{roc}, precision @ rank n:{prn}, AP:{ap}, \
              execution time: {duration}s'.format(
                clf_name=clf_name, roc=roc, prn=prn, ap=ap, duration=duration))

            time_mat[i, classifiers_indices[clf_name]] = duration
            roc_mat[i, classifiers_indices[clf_name]] = roc
            prn_mat[i, classifiers_indices[clf_name]] = prn
            ap_mat[i, classifiers_indices[clf_name]] = ap

    time_list = time_list + np.mean(time_mat, axis=0).tolist()
    temp_df = pd.DataFrame(time_list).transpose()
    temp_df.columns = df_columns
    time_df = pd.concat([time_df, temp_df], axis=0)

    roc_list = roc_list + np.mean(roc_mat, axis=0).tolist()
    temp_df = pd.DataFrame(roc_list).transpose()
    temp_df.columns = df_columns
    roc_df = pd.concat([roc_df, temp_df], axis=0)

    prn_list = prn_list + np.mean(prn_mat, axis=0).tolist()
    temp_df = pd.DataFrame(prn_list).transpose()
    temp_df.columns = df_columns
    prn_df = pd.concat([prn_df, temp_df], axis=0)

    ap_list = ap_list + np.mean(ap_mat, axis=0).tolist()
    temp_df = pd.DataFrame(ap_list).transpose()
    temp_df.columns = df_columns
    ap_df = pd.concat([ap_df, temp_df], axis=0)

    # Save the results for each run
    time_df.to_csv('time.csv', index=False, float_format='%.3f')
    roc_df.to_csv('roc.csv', index=False, float_format='%.3f')
    prn_df.to_csv('prc.csv', index=False, float_format='%.3f')
    ap_df.to_csv('ap.csv', index=False, float_format='%.3f')