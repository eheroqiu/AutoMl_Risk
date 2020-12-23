
# coding : uft-8

"""
XG boost的自动化炼丹炉


检查依赖
此文件初始化内部项目及 标准库(os, sys, warning )开源内容( XG boost, sk-learn, sko )导入
以及确认各开源内容的版本
    
"""


try:
    import sklearn
    from xgboost.sklearn import XGBClassifier
    import geatpy
except ImportError:
    pass

