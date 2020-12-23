

from gplearn.genetic import SymbolicTransformer
from sklearn.utils import check_random_state
import numpy as np
import pandas as pd
function_set = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg', 'inv',
                'max', 'min']
# 读取配置文件以支持更多越更少的超参数列表
import configparser
config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")


class Transformer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.transformer = SymbolicTransformer(generations=config.getint('population', 'MAXGEN'), population_size=2000,
                         hall_of_fame=100, n_components=10,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=1,
                         random_state=0, n_jobs=-1)

    def transform_feature(self):
        self.transformer.fit(self.X, self.y)
        self.gp_features = self.transformer.transform(self.X)
        self.new_features = np.hstack((self.X, self.gp_features))
        return self.new_features


if __name__ == "__main__":
    X_scaled, raw_data_y = pd.read_csv('demo_lib/xgboost_auto/data_test/creditcard.csv').iloc[:, 1:-1], \
                           pd.Series(pd.read_csv('demo_lib/xgboost_auto/data_test/creditcard.csv').iloc[:, -1])
    # 测试用例
    data = X_scaled
    data_t = Transformer(data, raw_data_y).transform_feature()
    #### 测试特征衍生前后的评分差距
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import scale
    est, est_t = LogisticRegression(penalty='l2', max_iter=10000), LogisticRegression(penalty='l2', max_iter=10000)
    # est.fit(data, raw_data_y)
    # est_t.fit(data_t, raw_data_y)
    print("特征衍生前的模型能力", (cross_val_score(est, scale(data), raw_data_y, cv=10, scoring='roc_auc')).mean())
    print("特征衍生后的模型能力", (cross_val_score(est_t, scale(data_t), raw_data_y, cv=10, scoring='roc_auc')).mean())





