from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
import numpy as np
import datetime
import math

def MaxDrawdown(series):
    # 计算最大回撤
    drawdown=np.zeros(len(series))
    tuple_lst=[]
    for i in range(len(series)-1):
        tuple_lst.append((series.iloc[i]-series.iloc[i+1:].min(),series.iloc[i]))
    max_drawdown=-100
    max_drawdown_rate=0
    for t in tuple_lst:
        if t[0] > max_drawdown:
            max_drawdown=t[0]
            max_drawdown_rate=t[0]/t[1]
    return max_drawdown_rate

def summary(context):
    # 可视化
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    factor_name=context.Path_factor.split('/')[-1].split('.')[0]


    with PdfPages('./result/'+factor_name+'_'+str(context.freq)+'.pdf') as pdf:
        # 分组回测
        fig=plt.figure(figsize=(10, 6))
        plt.bar(context.net_value.columns,context.net_value.iloc[-1,:]-1,color=10*['cyan']+['silver'])
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
        plt.grid(axis="y")
        plt.xticks(rotation=-45)
        plt.title('分组回测')
        pdf.savefig(fig)
        plt.close()

        # 多空净值
        fig=plt.figure(figsize=(10, 6))
        LS=context.net_value['group 10']-context.net_value['group 1']
        td_num=len(context.trade_day)
        Drawdown=MaxDrawdown(1+LS)   # 最大回撤
        year_return=(1+LS[context.end_date])**(252/td_num)-1 # 年化收益率
        sigma=(1+LS).pct_change().std(skipna=True)*math.sqrt(252)
        SR=(year_return-0.03)/sigma # 夏普比
        title='总收益率:'+'{:.2%}'.format(LS[context.end_date])+\
            '   年化收益率:'+'{:.2%}'.format(year_return)+\
            '   波动率:'+'{:.2%}'.format(sigma)+\
            '   夏普比率:'+'{:.2f}'.format(SR)+\
            '   最大回撤:'+'{:.2%}'.format(Drawdown)
        plt.plot(LS,label='long-short')
        plt.plot(context.net_value['benchmark']-1,label='benchmark')
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
        plt.grid(axis="y")
        plt.xticks(rotation=-45)
        plt.legend()
        plt.title('多空净值\n\n'+title)
        pdf.savefig(fig)
        plt.close()

        # 超额收益
        fig=plt.figure(figsize=(10, 6))
        Excess_return=context.net_value['group 10']-context.net_value['group 1']-context.net_value['benchmark']+1
        td_num=len(context.trade_day)
        Drawdown=MaxDrawdown(1+Excess_return)   # 最大回撤
        year_return=(1+Excess_return[context.end_date])**(252/td_num)-1 # 年化收益率
        sigma=(1+Excess_return).pct_change().std(skipna=True)*math.sqrt(252)
        SR=(year_return-0.03)/sigma # 夏普比
        title='总收益率:'+'{:.2%}'.format(LS[context.end_date])+\
            '   年化收益率:'+'{:.2%}'.format(year_return)+\
            '   波动率:'+'{:.2%}'.format(sigma)+\
            '   夏普比率:'+'{:.2f}'.format(SR)+\
            '   最大回撤:'+'{:.2%}'.format(Drawdown)
        plt.plot(Excess_return,label='Excess_return')
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
        plt.grid(axis="y")
        plt.xticks(rotation=-45)
        plt.legend()
        plt.title('超额收益\n\n'+title)
        pdf.savefig(fig)
        plt.close()

        # IC, Rank_IC
        fig=plt.figure(figsize=(10, 6))
        ICIR=np.mean(context.history['IC'])/np.std(context.history['IC'], ddof=1)
        Rank_ICIR=np.mean(context.history['Rank_IC'])/np.std(context.history['Rank_IC'], ddof=1)
        title='IC: 均值:'+'{:.2%}'.format(np.mean(context.history['IC']))+\
            '   最大值:'+'{:.2%}'.format(np.max(context.history['IC']))+\
            '   最小值:'+'{:.2%}'.format(np.min(context.history['IC']))+\
            '   标准差:'+'{:.2%}'.format(np.std(context.history['IC'], ddof=1))+\
            '   IR:'+'{:.2f}'.format(ICIR)+\
            '   T值:'+'{:.2f}'.format(ICIR*np.sqrt(context.history['times']-2))+'\n\n'+\
            'Rank_IC: 均值:'+'{:.2%}'.format(np.mean(context.history['Rank_IC']))+\
            '   最大值:'+'{:.2%}'.format(np.max(context.history['Rank_IC']))+\
            '   最小值:'+'{:.2%}'.format(np.min(context.history['Rank_IC']))+\
            '   标准差:'+'{:.2%}'.format(np.std(context.history['Rank_IC'], ddof=1))+\
            '   IR:'+'{:.2f}'.format(Rank_ICIR)+\
            '   T值:'+'{:.2f}'.format(Rank_ICIR*np.sqrt(context.history['times']-2))
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
        plt.title('IC序列\n\n'+title)
        pdf.savefig(fig)
        plt.close()
