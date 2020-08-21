import os
import sys
import numpy as np
from time import time

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
from models.cod import COD

for dim in [10, 100, 1000, 10000]:
    for row in [1000, 10000, 100000, 1000000]:
        X = np.random.rand(row, dim)
        clf = COD()
        t0 = time()
        test_scores = clf.decision_function(X)
        t1 = time()
        duration = round(t1 - t0, ndigits=4)
        print('Dim: ', dim, 'Row: ', row, duration)