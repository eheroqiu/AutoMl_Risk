
import json
import time
from demo_lib.xgboost_auto.scoring.eval import Eval
from demo_lib.xgboost_auto.optimize.opt_ctr import OptHyper
from demo_lib.xgboost_auto.classifier.clr_ctr import model_ctr
from demo_lib.xgboost_auto.transformer.feature_transformer import Transformer

def get_data(params_code, X, y):
    # 此处应校验一下X以及y的合法性

    kwargs = json.loads(params_code)
    kwargs['X'] = X
    kwargs['y'] = y
    return kwargs

def feature_derive(**kwargs):
    data = kwargs['X']
    data_t = Transformer(data, kwargs['y']).transform_feature()
    #### 测试特征衍生前后的评分差距
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import scale
    est, est_t = LogisticRegression(penalty='l2', max_iter=10000), LogisticRegression(penalty='l2', max_iter=10000)
    # est.fit(data, raw_data_y)
    # est_t.fit(data_t, raw_data_y)
    print("特征衍生前的模型能力", cross_val_score(est, scale(data), kwargs['y'], cv=10, scoring='roc_auc').mean())
    print("特征衍生后的模型能力", cross_val_score(est_t, scale(data_t), kwargs['y'], cv=10, scoring='roc_auc').mean())
    kwargs['X'] = data_t
    print("特征衍生后增加了" + str(kwargs['X'].shape[1] - data.shape[1]) + "个新的特征")
    return kwargs

# 入参接口封装
def params_filter():

    pass


def train_beta(**kwargs):
    cost_time = 0
    # pre model
    clf = model_ctr(**kwargs).create_model()
    # clf.set_params(**(OptHyper(clf=clf, **kwargs).get_params()))
    if kwargs['hyper_params_mode'] == 'Y':
        # 如果没有给出参数列表， 则需要调用参数优化工具OptHyper
        # try:
        begin = time.time()
        clf.set_params(**(OptHyper(clf=clf, **kwargs).get_params()))
        cost_time = format(time.time() - begin, '.1f')
        # except:
        #     print('params model_type is wrong!!!')
        #     exit()
    key = Eval().model_eval(clf=clf, X=kwargs['X'], y=kwargs['y'], cv=kwargs['cv'], scoring=kwargs['score_type'])
    # print('搜索到最优的' + 'KS' + '均值是' + str(key))
    log = json.dumps({'code': 1, 'time': cost_time, 'score_type': kwargs['score_type'], 'score_value': key,
                      'params': clf.get_params()})
    print(log)
    #
