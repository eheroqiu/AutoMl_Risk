
from demo_lib.xgboost_auto.classifier.xgb import XgBoost
from demo_lib.xgboost_auto.classifier.gbm import Gbm
from sklearn.ensemble import GradientBoostingClassifier


class model_ctr:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs['model_type']

    def create_model(self):
        if self.model == 'XGBOOST':
            return self.xgb()
        elif self.model == 'GBDT':
            return self.gbm()
        else:
            raise

    def xgb(self):
        return XgBoost(**self.kwargs['params'])

    def gbm(self):
        # return Gbm(**self.kwargs['params'])

        return GradientBoostingClassifier(**self.kwargs['params'])
