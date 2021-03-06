from datetime import datetime,timedelta
from multiprocessing import Pool
from Summary import *
import pandas as pd
import numpy as np
import dateutil
import time
import os

class Context:
    def __init__(self, start_date, end_date, group_num, frequency, Path_factor):
        self.start_date=start_date  # 回测区间
        self.end_date=end_date
        self.td=None    # 此时日期：%Y-%m-%d
        self.N=None  # 资产数目
        self.group_num=group_num # 因子组数
        self.freq=frequency # 调仓频率:日频'd',周频'w',月频'm',财报公布期(5,9,11月)'f',每n个交易日n(int)
        self.pos_matrix=None   # 仓位矩阵
        self.net_value=None # 各组+基准净值
        self.net_value_left=None    # 当前各组合剩余净值
        self.last_td_mark=None
        self.history={}    # 历史换仓数据：换仓时点、换仓次数、IC、Rank_IC、因子值、价格、股票池、仓位

        self.Path_trade_day=None
        self.trade_day=None # 交易日数据
        self.Path_factor=Path_factor
        self.factor=None # 因子数据
        self.Path_price=None
        self.price=None # 股票后复权价格数据
        self.Path_ST=None
        self.ST=None # 股票是否ST或退市，是为1，否为0
        self.Path_suspension=None
        self.suspension=None # 股票是否停复牌，是为1，否为0
        self.Path_over1year=None
        self.over1year=None # 股票是否上市超过1年，是为1，否为0

def initialize(context):
    # 读取信息
    context.Path_trade_day="D:/study/compile/research/data/financial_data/trade_day.csv"
    context.Path_price='D:/study/compile/research/data/IndustryIndexFactor-收盘价.csv'
    # 交易日信息
    context.trade_day=pd.read_csv(context.Path_trade_day,parse_dates=['datetime'],index_col=['datetime'])
    context.trade_day=context.trade_day[context.start_date:context.end_date].index
    # 行业指数信息
    context.price=pd.read_csv(context.Path_price,index_col=0,parse_dates=[0])
    context.price=context.price[context.start_date:context.end_date]
    context.price.columns=[x for x in range(1,31)]
    context.price.drop(30,axis=1, inplace=True) # 去掉综合金融行业的数据

    # 因子信息，“日期*行业”形式
    context.factor=pd.read_csv(context.Path_factor,parse_dates=[0],index_col=[0])
    context.factor=context.factor[context.start_date:context.end_date]
        
    if isinstance(context.freq,int):
        context.last_td_mark=0
    context.N=context.factor.shape[1]
    # 净值矩阵初始化
    group_col=['group '+str(i+1) for i in range(context.group_num)]
    group_col.append('benchmark')
    context.net_value=pd.DataFrame(index=context.trade_day,columns=group_col)
    context.net_value.iloc[0,:]=1   # 设置初始各组资产为1
    # 历史换仓数据
    context.history={'td':[],'times':0,'IC':[],'Rank_IC':[],'factor':[],'price':[],'position':[]}

def rebalance(context):
    # 用均值填充NaN
    f=context.factor.loc[context.td,:]
    f[pd.isna(f)]=f.mean()
    f_rank=f.rank(method='first').values   # 使用rank排序，防止组间分布不均
    # 计算权重矩阵
    context.pos_matrix=np.zeros((context.N,context.group_num+1))
    for g in range(context.group_num):
        V_min=np.percentile(f_rank,100*g/context.group_num,interpolation='linear')
        V_max=np.percentile(f_rank,100*(g+1)/context.group_num,interpolation='linear')
        if g+1 == context.group_num:
            context.pos_matrix[:,g][(f_rank>=V_min) & (f_rank<=V_max)]=context.net_value_left[g]
        else:
            context.pos_matrix[:,g][(f_rank>=V_min) & (f_rank<V_max)]=context.net_value_left[g]
    context.pos_matrix[:,context.group_num]=context.net_value_left[context.group_num]
    # 组内等权
    context.pos_matrix=context.pos_matrix/np.count_nonzero(context.pos_matrix,axis=0)
    # 每个行业的仓位=现金比例/股票价格
    for g in range(context.group_num+1):
        context.pos_matrix[:,g]=context.pos_matrix[:,g]/context.price.loc[context.td,:].values

    # 存储换仓数据
    context.history['td'].append(context.td)
    context.history['times']+=1
    context.history['factor'].append(f.values)
    context.history['price'].append(context.price.loc[context.td,:].values)
    context.history['position'].append(context.pos_matrix)
    # 计算上次换仓IC
    if context.last_td_mark:
        # 初次建仓不计算
        stock_return=(context.history['price'][-1]-context.history['price'][-2])/context.history['price'][-2]
        factor=context.history['factor'][-2]
        corr=pd.DataFrame([stock_return,factor]).T.corr(method='pearson')
        context.history['IC'].append(corr.loc[0,1])
        rank_corr=pd.DataFrame([stock_return,factor]).T.corr(method='spearman')
        context.history['Rank_IC'].append(rank_corr.loc[0,1])

def handle_data(context):
    if not context.last_td_mark:
        # 最初建仓
        context.net_value_left=context.net_value.loc[context.td,:].values
        rebalance(context)
    else:
        # 利用仓位矩阵计算净值
        td_price=context.price.loc[context.td,:].fillna(value=0)
        context.net_value.loc[context.td,:]=td_price.dot(context.pos_matrix)
        # 更新剩余净值
        context.net_value_left=context.net_value.loc[context.td,:].values

        rebalance_month=[5,9,11]
        # 调仓
        if isinstance(context.freq,int) and context.last_td_mark == context.freq:
            # 固定交易日换仓
            rebalance(context)
            context.last_td_mark=0
        elif context.freq == 'd':
            # 每日换仓
            rebalance(context)
        elif context.freq == 'w' and (context.td.strftime('%W') != context.last_td_mark):
            # 每周换仓
            rebalance(context)
        elif context.freq == 'm' and (context.td.month != context.last_td_mark):
            # 每月换仓
            rebalance(context)
        elif context.freq == 'f' and (context.td.month in rebalance_month and context.td.month != context.last_td_mark):
            # 5,9,11的第一个交易日换仓,更新仓位矩阵
            rebalance(context)

def run(context):
    initialize(context)
    for td in context.trade_day:
        context.td=td
        handle_data(context)
        # 更改标记，用于判断是否换仓
        if isinstance(context.freq,int):
            context.last_td_mark+=1
        elif context.freq == 'w':
            context.last_td_mark=td.strftime('%W')
        else:
            context.last_td_mark=td.month
    summary(context)



#file_path="***"
##file_list=os.listdir(file_path)
#file_list=['roe.csv',]
#for f in file_list:
#    context=Context('20100101', '20210930', 10, 'f', file_path+f)
#    run(context)

path="***"
context=Context('20100101', '20210930', 10, 'm', path)
run(context)
