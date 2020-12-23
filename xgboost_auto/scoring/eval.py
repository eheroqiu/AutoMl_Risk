""""
此文件用于返回 每个待交叉验证后的所有评分均值
返回值 也是 寻优工具的目标函数aim
本版本暂不满足多评分优化

"""

from sklearn.metrics import SCORERS
from demo_lib.xgboost_auto.scoring import score_func
from sklearn.model_selection import cross_val_score
from sklearn.metrics.scorer import check_scoring


class Eval:
    def __init__(self):
        self.score_sk = SCORERS.keys()
        self.my_score = score_func.my_score

    def model_eval(self, clf, X, y, cv=5, scoring='roc_auc'):
        """

        :param clf: 分类器类型
        :param X: 观测变量X
        :param y: 目标变量y
        :param cv: 交叉验证折数
        :param scoring:  评分函数
        :return:

        """

        if scoring in self.score_sk:
            return (cross_val_score(clf, X, y, cv=cv, scoring=scoring)).mean()
        else:
            try:
                return (cross_val_score(clf, X, y, cv=cv, scoring=self.my_score[scoring])).mean()
            except:
                print('wrong score func type')
                pass

    def relax_eval(self, clf, X_train, y_train, X_test, y_test, scoring='roc_auc'):
        clf.fit(X_train,y_train)
        if scoring in self.score_sk:
            scorer = check_scoring(clf, scoring=scoring)
            return scorer(clf, X_test, y_test)
        else:
            try:
                scorer = self.my_score[scoring]
                return scorer(clf, X_test, y_test)
            except:
                print('Wrong Score-function Type')
                pass

