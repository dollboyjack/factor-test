from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from matplotlib import ticker
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
        self.history={}    # 历史换仓数据：换仓时点、换仓次数、IC、因子值、价格、股票池、仓位

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
    context.Path_trade_day="./data/trade_day.csv"
    context.Path_price="./data/ElementaryFactor-复权收盘价.csv"
    context.Path_ST="./data/ElementaryFactor-ST.csv"
    context.Path_suspension="./data/ElementaryFactor-停复牌.csv"
    context.Path_over1year="./data/ElementaryFactor-上市超过一年.csv"
    # 交易日信息
    context.trade_day=pd.read_csv(context.Path_trade_day,parse_dates=['datetime'],index_col=['datetime'])
    context.trade_day=context.trade_day[context.start_date:context.end_date].index
    # 股价信息
    context.price=pd.read_csv(context.Path_price,parse_dates=['datetime'],index_col=['datetime'])
    context.price=context.price[context.start_date:context.end_date]
    # ST股
    context.ST=pd.read_csv(context.Path_ST,parse_dates=['datetime'],index_col=['datetime'])
    context.ST=context.ST[context.start_date:context.end_date]
    # 是否停牌
    context.suspension=pd.read_csv(context.Path_suspension,parse_dates=['datetime'],index_col=['datetime'])
    context.suspension=context.suspension[context.start_date:context.end_date]
    # 上市满一年
    context.over1year=pd.read_csv(context.Path_over1year,parse_dates=['datetime'],index_col=['datetime'])
    context.over1year=context.over1year[context.start_date:context.end_date]

    # 因子信息，“日期*code”形式
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
    context.history={'td':[],'times':0,'IC':[],'factor':[],'price':[],'tradable':[],'position':[]}

def rebalance(context):
    # 筛选股票池，去掉ST、上市不满一年、不在市的股票
    td_ST=context.ST.loc[context.td,:].values
    td_suspension=context.suspension.loc[context.td,:].values
    td_over1year=context.over1year.loc[context.td,:].values
    # 得到可交易股票的矩阵
    tradable_matrix=(1-td_ST)*(1-td_suspension)*td_over1year
    # 用在市公司因子值的均值填充NaN
    f=context.factor.loc[context.td,:]
    f_rank=f.rank(method='first').values   # 使用rank排序，防止组间分布不均
    f_value=f_rank[tradable_matrix==1]
    f[np.isnan(f_rank)]=np.nanmean(f.values)
    f_rank[np.isnan(f_rank)]=np.nanmean(f_value)
    f_value=f_rank[tradable_matrix==1]   # 获取更新后的可交易因子值
    # 计算权重矩阵
    context.pos_matrix=np.zeros((context.N,context.group_num+1))
    for g in range(context.group_num):
        V_min=np.percentile(f_value,100*g/context.group_num,interpolation='linear')
        V_max=np.percentile(f_value,100*(g+1)/context.group_num,interpolation='linear')
        if g+1 == context.group_num:
            context.pos_matrix[:,g][(f_rank>=V_min) & (f_rank<=V_max)]=context.net_value_left[g]
        else:
            context.pos_matrix[:,g][(f_rank>=V_min) & (f_rank<V_max)]=context.net_value_left[g]
    context.pos_matrix[:,context.group_num][tradable_matrix==1]=context.net_value_left[context.group_num]
    # 去掉不在市的权重
    context.pos_matrix=context.pos_matrix*tradable_matrix.reshape([context.N,1])
    # 组内等权
    context.pos_matrix=context.pos_matrix/np.count_nonzero(context.pos_matrix,axis=0)
    # 每只股票的仓位=现金比例/股票价格
    for g in range(context.group_num+1):
        context.pos_matrix[:,g]=context.pos_matrix[:,g]/context.price.loc[context.td,:].values
    context.pos_matrix[np.isnan(context.pos_matrix)]=0

    # 存储换仓数据
    context.history['td'].append(context.td)
    context.history['times']+=1
    context.history['factor'].append(f.values)
    context.history['price'].append(context.price.loc[context.td,:].values)
    context.history['tradable'].append(tradable_matrix)
    context.history['position'].append(context.pos_matrix)
    # 计算上次换仓IC
    if context.last_td_mark:
        # 初次建仓不计算
        stock_return=(context.history['price'][-1]-context.history['price'][-2])/context.history['price'][-2]
        stock_return=stock_return[context.history['tradable'][-2]==1]
        stock_return[np.isnan(stock_return)]=0  # i或i-1任一期价格不存在设定收益率为0
        factor=context.history['factor'][-2][context.history['tradable'][-2]==1]
        corr=np.corrcoef(stock_return,factor)
        context.history['IC'].append(corr[0,1])

def MaxDrawdown(series):
    # 计算最大回撤
    drawdown=np.zeros(len(series))
    for i in range(len(series)-1):
        drawdown[i]=(series.iloc[i]-series.iloc[i+1:].min())/series.iloc[i]
    return drawdown.max()

def summary(context):
    # 可视化
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    ICIR=np.mean(context.history['IC'])/np.std(context.history['IC'], ddof=1)
    title='IC均值:'+'{:.2%}'.format(np.mean(context.history['IC']))+\
        '   IC最大值:'+'{:.2%}'.format(np.max(context.history['IC']))+\
        '   IC最小值:'+'{:.2%}'.format(np.min(context.history['IC']))+\
        '   IC标准差:'+'{:.2%}'.format(np.std(context.history['IC'], ddof=1))+\
        '   ICIR:'+'{:.2f}'.format(ICIR)+\
        '   T统计量:'+'{:.2f}'.format(ICIR*np.sqrt(context.history['times']-2))
    factor_name=context.Path_factor.split('/')[-1].split('.')[0]
    # IC图
    fig = plt.figure(figsize=(12, 9), dpi=100)
    x_label=[t.strftime('%Y-%m-%d') for t in context.history['td'][:-1]]
    plt.bar(x_label,context.history['IC'])
    if isinstance(context.freq,int):
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(252/(2*context.freq))))
    elif context.freq == 'd':
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(126))
    elif context.freq == 'w':
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(26))
    elif context.freq == 'm':
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(6))
    elif context.freq == 'f':
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
    plt.grid(axis="y")
    plt.xticks(rotation=-45)
    plt.title(title)
    #plt.show()
    plt.savefig('./result/IC_'+factor_name+'_'+str(context.freq)+'.png')

    # 多空图
    LS=context.net_value['group 10']-context.net_value['group 1']
    td_num=len(context.trade_day)
    Drawdown=MaxDrawdown(1+LS)   # 最大回撤
    year_return=(1+LS[context.end_date])**(252/td_num)-1 # 年化收益率
    title='多空总收益率:'+'{:.2%}'.format(LS[context.end_date])+\
        '   年化收益率:'+'{:.2%}'.format(year_return)+\
        '   最大回撤:'+'{:.2%}'.format(Drawdown)
    fig = plt.figure(figsize=(12, 9), dpi=100)    
    ax = fig.add_subplot(2, 1, 1)
    ax.bar(context.net_value.columns,context.net_value.iloc[-1,:]-1,color=10*['cyan']+['silver'])
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
    plt.grid(axis="y")
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(LS,label='long-short')
    ax.plot(context.net_value['benchmark']-1,label='benchmark')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
    plt.grid(axis="y")
    plt.xticks(rotation=-45)
    ax.legend()
    fig.suptitle(title)
    #plt.show()
    plt.savefig('./result/L-S_'+factor_name+'_'+str(context.freq)+'.png')

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



file_path="..."
file_list=os.listdir(file_path)
#file_list=[]
for f in file_list:
    context=Context('20170101', '20210630', 10, 10, file_path+f)
    run(context)
