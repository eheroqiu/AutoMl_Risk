

from demo_lib.xgboost_auto.optimize.grid import Grid
from demo_lib.xgboost_auto.optimize.evo import Evo


class OptHyper:
    def __init__(self, clf, **kwargs):
        self.clf = clf
        self.kwargs = kwargs
        self.Opt = kwargs['search_opt']

    def get_params(self):
        if self.Opt == 'GridCV':
            return self.grid_cv()
        elif self.Opt == 'Evol':
            return self.evol_ga()

    def grid_cv(self):
        return Grid(self.clf, **self.kwargs).get_params()

    def evol_ga(self):
        return Evo(self.clf, **self.kwargs).get_params()
