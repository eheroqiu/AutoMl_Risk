
from scoop import futures
import copy

import numpy as np
import geatpy as ea
import os
from demo_lib.xgboost_auto.scoring.eval import Eval
from sklearn.model_selection import train_test_split
import time
# 读取配置文件以支持更多越更少的超参数列表
import configparser
config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")



class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, clf, **kwargs):
        name = 'Evo find Hyper params'  # 初始化name（函数名称，可以随意设置）
        # self.pool = ThreadPool(6)
        # pool = Pool(6)
        self.clf = clf
        self.X = kwargs['X']
        self.y = kwargs['y']
        self.cv = kwargs['cv']
        self.scoring = kwargs['score_type']
        self.meta = [_ for _ in kwargs['hyper_params_meta']]
        self.relax = kwargs['use_relax']


        M = 1  # 初始化M（目标维数）
        maxormins = [config.getint(self.scoring, 'type')]  # 初始化 maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        # Dim = 3  # 初始化Dim（决策变量维数）
        Dim = len(kwargs['hyper_params_meta'])

        varTypes = []  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb, ub = [], []
        for i in kwargs['hyper_params_meta'].keys():
            x = kwargs['hyper_params_meta'][i]['lb']
            lb.append(x)
            y = kwargs['hyper_params_meta'][i]['ub']
            ub.append(y)
            content = eval(config.get(kwargs['model_type'], i))
            varTypes.append(int(content['type']))
        lbin = [1] * Dim  # 决策变量下边界
        ubin = [1] * Dim  # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def nind_eval(self, i):
        kwargs = dict()
        value = i[1]
        for j in range(i[2]):
            kwargs[self.meta[j]] = (int(value[j]) if np.ceil(value[j]) == np.floor(value[j]) else round(value[j], 2))
        clf = copy.copy(self.clf)
        clf.set_params(**kwargs)
        """使用relax方法 则不使用交叉验证的方式来评估超参的效果评估，而选用 1.代内固定训练验证集 2. 代间随机拆分训练测试的方法来加速进化过程"""
        if self.relax == 1:
            tmp = Eval().relax_eval(clf=clf, X_train=i[3], y_train=i[5], X_test=i[4], y_test=i[6], scoring=self.scoring)
        else:
            tmp = Eval().model_eval(clf=clf, X=self.X, y=self.y, cv=self.cv,
                                scoring=self.scoring)
        return round(tmp, 3)

    def aimFunc(self, pop):  # 目标函数

        begin = time.time()
        Vars = pop.Phen  # 得到决策变量矩阵
        size = [len(self.meta)] * pop.sizes
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=0.3)
        X_train = [X_train] * pop.sizes
        X_test = [X_test] * pop.sizes
        y_train = [y_train] * pop.sizes
        y_test = [y_test] * pop.sizes
        # pool = Pool(6)

        args = zip(range(len(Vars)), Vars, size, X_train, X_test, y_train, y_test)

        #args = [(i, size) for i in zip(range(len(Vars)), Vars)]

        x = futures.map(self.nind_eval, [i for i in args])
        ObjV2 = list(x)
        print(self.scoring, max(ObjV2))
        print('本轮进化计算耗时：', time.time()- begin)
        '''for i in sorted(x):
            ObjV2.append(i[1])'''
        pop.ObjV = np.array([ObjV2]).T



        # 用决策变量的矩阵的每一行元素去 赋值模型超参数
        # 需要对参数做一定程度的解析 拼接

        # 利用model_eval 函数去生成目标变量
        # 将决策变量和目标函数值都存入pop.objv中
        # aimFunc 自身不做任何数值的返回


class Evo:

    def __init__(self, clf, **kwargs):

        """================================实例化问题对象============================="""
        problem = MyProblem(clf, **kwargs)  # 生成问题对象


        """==================================种群设置================================"""
        Encoding = 'RI'  # 编码方式
        NIND = config.getint('population', 'NIND')  # 种群规模
        Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
        population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）

        """================================算法参数设置==============================="""
        myAlgorithm = ea.soea_SEGA_templet(problem, population)  # 实例化一个算法模板对象`
        myAlgorithm.MAXGEN = config.getint('population', 'MAXGEN')  # 最大进化代数
        myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制过程动画）
        """===========================调用算法模板进行种群进化===========================
        调用run执行算法模板，得到帕累托最优解集NDSet。NDSet是一个种群类Population的对象。
        NDSet.ObjV为最优解个体的目标函数值；NDSet.Phen为对应的决策变量值。
        详见Population.py中关于种群类的定义。
        """
        self.NDSet = myAlgorithm.run()[0]  # 执行算法模板，得到非支配种群

        params_lst = iter(kwargs['hyper_params_meta'].keys())
        # 最后一轮的种群变量
        ndset = self.NDSet.ObjV.tolist()
        # 拥有最大目标函数的超参数组合
        var_max = iter(np.array(self.NDSet.Phen[ndset.index(max(ndset))], dtype=object))

        self.params_evo = dict()
        for x in params_lst:
            value = next(var_max)
            self.params_evo[x] = (int(value) if np.ceil(value)==np.floor(value) else format(value, '.2f'))
        print(self.params_evo)

        '''problem.pool.close()
        problem.pool.join()'''
        '''
        字典还是不太好封装整数和浮点共存的数据结构
        print(np.array(var_max, dtype=object))
        x = pd.DataFrame(params_lst, columns={'params'})
        y = pd.DataFrame(np.array(var_max, dtype=object), columns={'value'})
        self.params_evo = dict(zip(x['params'], y['value']))
        '''

    def get_params(self):
        return self.params_evo
