from xgboost.sklearn import XGBClassifier


class XgBoost(XGBClassifier):
    def __init__(self, **kwargs):
        XGBClassifier.__init__(self, **kwargs)


'''if __name__ == "__main__":
    print(XgBoost(n_estimators=100,max_depth=3 , learning_rate=2).get_params())'''