from abc import ABC

from sklearn.ensemble import GradientBoostingClassifier


class Gbm(GradientBoostingClassifier):

    def __init__(self, **kwargs):
        GradientBoostingClassifier.__init__(self, **kwargs)