from __future__ import division
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from pyod.utils.utility import precision_n_scores
from pyod.utils.utility import standardizer
from pyod.models.sos import SOS
from pyod.models.sod import SOD
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.mcd import MCD
from pyod.models.lof import LOF
from pyod.models.loda import LODA
from pyod.models.lmdd import LMDD
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.cof import COF
from pyod.models.cblof import CBLOF
from pyod.models.abod import ABOD
import arff
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import os
import sys
from time import time
import pdb
# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
from models.cod_exp import COD


def read_arff(file_path, misplaced_list):
    misplaced = False
    for item in misplaced_list:
        if item in file_path:
            misplaced = True

    file = arff.load(open(file_path))
    data_value = np.asarray(file['data'])
    attributes = file['attributes']

    X = data_value[:, 0:-2]
    if not misplaced:
        y = data_value[:, -1]
    else:
        y = data_value[:, -2]
    y[y == 'no'] = 0
    y[y == 'yes'] = 1
    y = y.astype('float').astype('int').ravel()

    if y.sum() > len(y):
        print(attributes)
        raise ValueError('wrong sum')

    return X, y, attributes


# Define data file and read X and y
file_names = [
    'Arrhythmia',
    'Cardiotocography',
    'HeartDisease',
    'Hepatitis',
    'InternetAds',
    'Ionosphere',
    'KDDCup99',
    'Lymphography',
    'Pima',
    'Shuttle',
    'SpamBase',
    'Stamps',
    'Waveform',
    'WBC',
    'WDBC',
    'WPBC',
]

#############################################################################
misplaced_list = ['Arrhythmia', 'Cardiotocography', 'Hepatitis', 'ALOI',
                  'KDDCup99']
arff_list = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data',
                                 'semantic', 'Arrhythmia', 'Arrhythmia_withoutdupl_46.arff')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'semantic', 'Cardiotocography',
                                 'Cardiotocography_withoutdupl_22.arff')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'semantic', 'HeartDisease',
                                 'HeartDisease_withoutdupl_44.arff')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data',
                                 'semantic', 'Hepatitis', 'Hepatitis_withoutdupl_16.arff')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'semantic', 'InternetAds',
                                 'InternetAds_withoutdupl_norm_19.arff')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data',
                                 'literature', 'Ionosphere', 'Ionosphere_withoutdupl_norm.arff')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                 'data', 'literature', 'KDDCup99', 'KDDCup99_idf.arff')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data',
                                 'literature', 'Lymphography', 'Lymphography_withoutdupl_idf.arff')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                 'data', 'semantic', 'Pima', 'Pima_withoutdupl_35.arff')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data',
                                 'literature', 'Shuttle', 'Shuttle_withoutdupl_v01.arff')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                 'data', 'semantic', 'SpamBase', 'SpamBase_withoutdupl_40.arff')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                 'data', 'semantic', 'Stamps', 'Stamps_withoutdupl_09.arff')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data',
                                 'literature', 'Waveform', 'Waveform_withoutdupl_v01.arff')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                 'data', 'literature', 'WBC', 'WBC_withoutdupl_v01.arff')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                 'data', 'literature', 'WDBC', 'WDBC_withoutdupl_v01.arff')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                 'data', 'literature', 'WPBC', 'WPBC_withoutdupl_norm.arff'))
]

n_ite = 10
n_classifiers = 6

df_columns = ['Data', '# Samples', '# Dimensions', 'Outlier Perc',
              'COD_L', 'COD_R', 'COD_B', 'COD_S', 'COD_M', 'COD']

# initialize the container for saving the results
roc_df = pd.DataFrame(columns=df_columns)
prn_df = pd.DataFrame(columns=df_columns)
ap_df = pd.DataFrame(columns=df_columns)
time_df = pd.DataFrame(columns=df_columns)

for j in range(len(arff_list)):
    mat_file = file_names[j]
    mat_file_path = arff_list[j]
    X, y, attributes = read_arff(mat_file_path, misplaced_list)

    outliers_fraction = np.count_nonzero(y) / len(y)
    outliers_percentage = round(outliers_fraction * 100, ndigits=4)

    # construct containers for saving results
    roc_list = [mat_file, X.shape[0], X.shape[1], outliers_percentage]
    prn_list = [mat_file, X.shape[0], X.shape[1], outliers_percentage]
    ap_list = [mat_file, X.shape[0], X.shape[1], outliers_percentage]
    time_list = [mat_file, X.shape[0], X.shape[1], outliers_percentage]

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

            print('{clf_name} ROC:{roc}, precision @ rank n:{prn}, AP:{ap},'
                  'execution time: {duration}s'.format(
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
