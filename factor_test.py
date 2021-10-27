from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
import numpy as np
import dateutil
import time

class Context:
    def __init__(self, f_start_date, start_date, end_date, group_num):
        self.f_start_date=f_start_date  # 因子开始时间（比如财报1月1日只能用去年的数据，但是因子时间为报告期）
        self.start_date=start_date  # 回测区间
        self.end_date=end_date
        self.td=None    # 此时日期：%Y-%m-%d
        self.N=None  # 资产数目
        self.group_num=group_num # 因子组数
        self.pos_matrix=None   # 仓位矩阵
        self.net_value=None # 各组+基准净值

        self.Path_trade_day=None
        self.trade_day=None # 交易日数据
        self.Path_factor=None
        self.factor=None # 因子数据
        self.Path_price=None
        self.price=None # 股票后复权价格数据
        self.Path_tradable=None
        self.tradable=None # 股票是否在市数据

def initialize(context):
    # 读取信息
    context.Path_trade_day="./data/trade_day.csv"
    context.Path_price="./data/ElementaryFactor-复权收盘价.csv"
    context.Path_tradable="./data/ElementaryFactor-是否在市.csv"
    context.Path_factor="D:/study/compile/research/factor_database/corporation_factor/profit_factor/oper_rev2assets.csv"
    # 交易日信息
    context.trade_day=pd.read_csv(context.Path_trade_day,parse_dates=['datetime'],index_col=['datetime'])
    context.trade_day=context.trade_day[context.start_date:context.end_date].index
    # 因子信息，“日期*code”形式
    context.factor=pd.read_csv(context.Path_factor,parse_dates=['datetime'],index_col=['datetime'])
    context.factor=context.factor[context.f_start_date:context.end_date]
    # 股价信息
    context.price=pd.read_csv(context.Path_price,parse_dates=['datetime'],index_col=['datetime'])
    context.price=context.price[context.start_date:context.end_date]
    # 股票是否在市
    context.tradable=pd.read_csv(context.Path_tradable,parse_dates=['datetime'],index_col=['datetime'])
    context.tradable=context.tradable[context.start_date:context.end_date]
    context.tradable=context.tradable.fillna(0)
    
    context.N=context.factor.shape[1]
    # 净值矩阵初始化
    group_col=['group '+str(i+1) for i in range(context.group_num)]
    group_col.append('benchmark')
    context.net_value=pd.DataFrame(index=context.trade_day,columns=group_col)
    context.net_value.iloc[0,:]=1   # 设置初始各组资产为1

def rebalance(context,net_value_left):
    # 计算使用报告期
    if context.td.month in [1,2,3,4]:
        # 使用去年三季报数据
        financial_time=datetime(context.td.year-1,9,30)
    elif context.td.month in [5,6,7,8]:
        # 使用一季报数据
        financial_time=datetime(context.td.year,3,31)
    elif context.td.month in [9,10]:
        # 使用半年报数据
        financial_time=datetime(context.td.year,6,30)
    elif context.td.month in [11,12]:
        # 使用三季报数据
        financial_time=datetime(context.td.year,9,30)
    # 计算权重矩阵
    f=context.factor.loc[financial_time,:].values
    f_value=f[~np.isnan(f)]
    context.pos_matrix=np.zeros((context.N,context.group_num+1))
    for g in range(context.group_num):
        V_min=np.percentile(f_value,100*g/context.group_num,interpolation='linear')
        V_max=np.percentile(f_value,100*(g+1)/context.group_num,interpolation='linear')
        if g+1 == context.group_num:
            context.pos_matrix[:,g][(f>=V_min) & (f<=V_max)]=net_value_left[g]
        else:
            context.pos_matrix[:,g][(f>=V_min) & (f<V_max)]=net_value_left[g]
    context.pos_matrix[:,context.group_num][~np.isnan(f)]=net_value_left[context.group_num]
    # 去掉不在市的权重
    tradable_matrix=context.tradable.loc[context.td,:].values
    context.pos_matrix=context.pos_matrix*tradable_matrix.reshape([context.N,1])
    # 组内等权
    context.pos_matrix=context.pos_matrix/np.count_nonzero(context.pos_matrix,axis=0)
    # 每只股票的仓位=现金比例/股票价格
    for g in range(context.group_num+1):
        context.pos_matrix[:,g]=context.pos_matrix[:,g]/context.price.loc[context.td,:].values
    context.pos_matrix[np.isnan(context.pos_matrix)]=0

def summary(context):
    # 可视化
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    fig = plt.figure(figsize=(12, 6), dpi=100)
    ax = fig.add_subplot(2, 1, 1)
    ax.bar(context.net_value.columns,context.net_value.iloc[-1,:]-1,color=10*['cyan']+['silver'])
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
    plt.grid(axis="y")
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(context.net_value['group 10']-context.net_value['group 1'],label='long-short')
    ax.plot(context.net_value['benchmark']-1,label='benchmark')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
    plt.grid(axis="y")
    plt.xticks(rotation=-45)
    ax.legend()
    #plt.show()
    outpath='./result/L-S_'+context.Path_factor.split('/')[-1].split('.')[0]+'.png'
    plt.savefig(outpath)

def handle_data(context, td_num, last_td_month):
    rebalance_month=[5,9,11]
    if td_num == 0:
        # 最初建仓
        net_value_left=context.net_value.iloc[td_num,:].values
        rebalance(context,net_value_left)
    else:
        if context.td.month in rebalance_month and context.td.month != last_td_month:
            # 5,9,11的第一个交易日换仓,更新仓位矩阵
            net_value_left=context.net_value.iloc[td_num-1,:].values
            rebalance(context,net_value_left)
        # 利用仓位矩阵计算净值
        td_price=context.price.loc[context.td,:].fillna(value=0)
        context.net_value.loc[context.td,:]=td_price.dot(context.pos_matrix)

def run(context):
    initialize(context)
    last_td_month=12 if int(context.start_date[4:6]) == 1 else int(context.start_date[4:6])-1
    td_num=-1
    for td in context.trade_day:
        td_num+=1
        context.td=td
        handle_data(context, td_num, last_td_month)
        last_td_month=td.month
    summary(context)

context=Context('20160630','20170101','20210630',10)
run(context)

#################################################################
'''
遗留问题：
1.在市但没有因子值的股票处理有点问题
2.IC,IR
3.日频、周频、月频的调仓补充
'''