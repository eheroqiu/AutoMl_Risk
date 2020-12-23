from demo_lib.xgboost_auto.scoring.eval import Eval
from sklearn.model_selection import RandomizedSearchCV
from numpy import arange
import numpy as np



class Random():

    def __init__(self, clf, **kwargs):
        score = kwargs['score_type']
        self.eval = Eval()
        params = dict()
        for i in kwargs['hyper_params_meta'].keys():
            x = kwargs['hyper_params_meta'][i]['lb']
            y = kwargs['hyper_params_meta'][i]['ub']
            z = kwargs['hyper_params_meta'][i]['step']
            params[i] = arange(x, y, z)
            print(params[i])
        self.rs = RandomizedSearchCV(estimator=clf,
                                 n_jobs=-1,
                                 param_grid=params,
                                 scoring=(score if score in self.eval.score_sk
                                          else self.eval.my_score[score]),
                                 cv=kwargs['cv'])
        self.X = kwargs['X']
        self.y = kwargs['y']
        self.rs.fit(self.X, self.y)
        for k, v in self.rs.best_params_.items():
            self.rs.best_params_[k]= (int(v) if np.ceil(v) == np.floor(v) else format(v, '.2f'))


    def get_params(self):
        print(self.rs.best_params_)
        return self.rs.best_params_


