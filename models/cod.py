from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from pyod.models.base import BaseDetector
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import skew
import matplotlib.pyplot as plt

class COPOD(BaseDetector):
    def __init__(self, contamination=0.1):
        super(COPOD, self).__init__(contamination=contamination)

    def ecdf(self, X):
        ecdf = ECDF(X)
        return ecdf(X)

    def fit(self, X, y=None):
        self.X_train = X

    def decision_function(self, X):
        if hasattr(self, 'X_train'):
            original_size = X.shape[0]
            X = np.concatenate((self.X_train, X), axis=0)
        size = X.shape[0]
        dim = X.shape[1]
        self.U_l = pd.DataFrame(-1*np.log(np.apply_along_axis(self.ecdf, 0, X)))
        self.U_r = pd.DataFrame(-1*np.log(np.apply_along_axis(self.ecdf, 0, -X)))
        skewness = np.sign(np.apply_along_axis(skew, 0, X))
        self.U_skew = self.U_l * -1*np.sign(skewness - 1) + self.U_r * np.sign(skewness + 1)
        self.O = np.maximum(self.U_skew, np.add(self.U_l, self.U_r)/2)
        if hasattr(self, 'X_train'):
            self.decision_scores_ = self.O.sum(axis=1).to_numpy()[-original_size:]
        else:
            self.decision_scores_ = self.O.sum(axis=1).to_numpy()
        self.threshold_ = np.percentile(self.decision_scores_, (1-self.contamination)*100)
        self.labels_ = np.zeros(len(self.decision_scores_))
        for i in range(len(self.decision_scores_)):
            self.labels_[i] = 1 if self.decision_scores_[i] >= self.threshold_ else 0
        return self.decision_scores_

    def explain_outlier(self, ind, cutoffs=None):
        cutoffs = [1-self.contamination, 0.99] if cutoffs is None else cutoffs
        plt.plot(range(1, self.O.shape[1] + 1), self.O.iloc[ind], label='Outlier Score')
        for i in cutoffs:
            plt.plot(range(1, self.O.shape[1] + 1), self.O.quantile(q=i, axis=0), '-', label=f'{i} Cutoff Band')
        plt.xlim([1, self.O.shape[1] + 1])
        plt.ylim([0, int(self.O.max().max()) + 1])
        plt.ylabel('Dimensional Outlier Score')
        plt.xlabel('Dimension')
        plt.xticks(range(1, self.O.shape[1] + 1))
        plt.yticks(range(0, int(self.O.max().max()) + 1))
        label = 'Outlier' if self.labels_[ind] == 1 else 'Inlier'
        plt.title(f'Outlier Score Breakdown for Data #{ind+1} ({label})')
        plt.legend()
        plt.show()
        return self.O.iloc[ind], self.O.quantile(q=cutoffs[0], axis=0), self.O.quantile(q=cutoffs[1], axis=0)