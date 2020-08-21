# COPOD
**Cop**ula-based **O**utlier **D**etection: a *fast*, *parameter-free*, and *highly interpretable* unsupervised outlier detection method.

-----


Li, Z., Zhao, Y., Botta, N., Ionescu, C. and Hu, X. COPOD: Copula-Based Outlier Detection. *IEEE International Conference on Data Mining (ICDM)*, 2020.

Please cite the paper as:

    @inproceedings{li2020copod,
      title={{COPOD:} Copula-Based Outlier Detection},
      author={Li, Zheng and Zhao, Yue and Botta, Nicola and Ionescu, Cezar and Hu, Xiyang},
      booktitle={IEEE International Conference on Data Mining (ICDM)},
      year={2020},
      organization={IEEE},
    }


[PDF for Personal Use (raw version)](http://localhost:1313/user/yuezhao2/papers/20-icdm-copod-preprint.pdf) | 
[Presentation Slides](https://github.com/winstonll/COPOD) | 
[API Documentation](https://github.com/winstonll/COPOD) | 
[Example with PyOD](https://github.com/winstonll/COPOD) 

------------

##  Introduction
Outlier detection refers to the identification of rare items that are deviant from the general data distribution. Existing unsupervised approaches suffer from high computational complexity, low predictive capability, and limited interpretability. As a remedy, we present a novel outlier detection algorithm called COPOD, which is inspired by statistical methods for modeling multivariate data distribution. COPOD first constructs the empirical copula, and then uses the fitted model to predict tail probabilities of each given data point to determine its level of “extremeness”. Intuitively, we think of this as calculating an anomalous p-value. This makes COPOD both parameter-free, highly interpretable, as well as computationally efficient. Moreover, COPOD is parameter-free and require no tuning, which reduces human subjectivity and bias. In this work, we make three key contributions, 1) propose a novel, parameter-free outlier detection algorithm with both great performance and interpretability, 2) perform extensive experiments on 30 benchmark datasets to show that COPOD outperforms in most cases, at the same time is also one of the fastest outlier detection algorithms, and 3) release an easy-to-use Python implementation for reproducibility.


## Dependency
The experiment codes are writen in Python 3.6 and built on a number of Python packages:
- numpy>=1.13
- numba>=0.35
- pyod
- scipy>=0.19
- scikit_learn>=0.19

Batch installation is possible using the supplied "requirements.txt" with pip or conda.

````cmd
pip install -r requirements.txt
````

## Reproducibility & Production Level Code

TBA here soon.



