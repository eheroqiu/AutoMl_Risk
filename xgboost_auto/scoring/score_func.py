"""
此文件用于新增所有 sklearn 非原生自带的scoring 办法

"""

from sklearn import metrics
from sklearn.metrics import make_scorer


# 计算ks值


def eval_ks_score(y_true, y_predicted):
    label = y_true
    fpr, tpr, _ = metrics.roc_curve(label, y_predicted, pos_label=1)
    return abs(fpr - tpr).max()


def eval_oth_score(y_true, y_predicted):
    # fake wrong
    return 0.5


my_score = {'KS': make_scorer(eval_ks_score, greater_is_better=True),
            'oth': make_scorer(eval_oth_score, greater_is_better=True)}
