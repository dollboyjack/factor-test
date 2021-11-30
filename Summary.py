from datetime import datetime,timedelta
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
import numpy as np
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
    result={}

    if np.mean(context.history['Rank_IC']) >= 0:
        #记录因子值与收益相关性
        relation=1
    else:
        relation=0
    # 计算多空和超额收益
    summary_df=pd.DataFrame(index=context.net_value.index,columns=['Long','Short','benchmark','LS','Excess'])
    summary_df['Long']=context.net_value['group '+str(context.group_num)] if relation else context.net_value['group 1']
    summary_df['Short']=context.net_value['group 1'] if relation else context.net_value['group '+str(context.group_num)]
    summary_df['benchmark']=context.net_value['benchmark']
    summary_df=summary_df.pct_change()
    summary_df['LS']=summary_df['Long']-summary_df['Short']
    summary_df['Excess']=summary_df['Long']-summary_df['benchmark']
    summary_df=(summary_df+1).cumprod()
    summary_df.iloc[0,:]=1


    with PdfPages('./result/'+factor_name+'_'+str(context.freq)+'.pdf') as pdf:
        # 分组回测
        fig=plt.figure(figsize=(10, 6))
        td_num=len(context.trade_day)
        return_year=context.net_value.iloc[-1,:]**(252/td_num)-1
        if relation == 0:
            # 使第一组为空头，第5组为多头
            return_year[:-1]=return_year[:-1][::-1]
        plt.bar(context.net_value.columns,return_year,color=context.group_num*['cyan']+['silver'])
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
        plt.grid(axis="y")
        plt.xticks(rotation=-45)
        plt.title('分组年化收益率')
        pdf.savefig(fig)
        plt.close()

        # 多空净值
        fig=plt.figure(figsize=(10, 6))
        LS=summary_df['LS']
        result['多空最大回撤']=MaxDrawdown(1+LS)   # 最大回撤
        result['多空年化收益率']=LS[context.end_date]**(252/td_num)-1 if (1+LS[context.end_date]) > 0 else np.nan
        result['多空年化波动率']=LS.pct_change().std(skipna=True)*math.sqrt(252)
        result['多空夏普比率']=(result['多空年化收益率']-0.03)/result['多空年化波动率'] # 夏普比
        title='总收益率:'+'{:.2%}'.format(LS[context.end_date]-1)+\
            '   年化收益率:'+'{:.2%}'.format(result['多空年化收益率'])+\
            '   波动率:'+'{:.2%}'.format(result['多空年化波动率'])+\
            '   夏普比率:'+'{:.2f}'.format(result['多空夏普比率'])+\
            '   最大回撤:'+'{:.2%}'.format(result['多空最大回撤'])
        plt.plot(LS-1,label='long-short')
        plt.plot(summary_df['benchmark']-1,label='benchmark')
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
        plt.grid(axis="y")
        plt.xticks(rotation=-45)
        plt.legend()
        plt.title('多空净值\n\n'+title)
        pdf.savefig(fig)
        plt.close()

        # 超额收益
        fig=plt.figure(figsize=(10, 6))
        Excess_return=summary_df['Excess']
        result['超额最大回撤']=MaxDrawdown(Excess_return)   # 最大回撤
        result['超额年化收益率']=Excess_return[context.end_date]**(252/td_num)-1 if (1+Excess_return[context.end_date]) > 0 else np.nan
        result['超额年化波动率']=Excess_return.pct_change().std(skipna=True)*math.sqrt(252)
        result['超额夏普比率']=(result['超额年化收益率']-0.03)/result['超额年化波动率'] # 夏普比
        title='总收益率:'+'{:.2%}'.format(Excess_return[context.end_date]-1)+\
            '   年化收益率:'+'{:.2%}'.format(result['超额年化收益率'])+\
            '   波动率:'+'{:.2%}'.format(result['超额年化波动率'])+\
            '   夏普比率:'+'{:.2f}'.format(result['超额夏普比率'])+\
            '   最大回撤:'+'{:.2%}'.format(result['超额最大回撤'])
        plt.plot(Excess_return-1,label='Excess_return')
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
        plt.grid(axis="y")
        plt.xticks(rotation=-45)
        plt.legend()
        plt.title('超额收益\n\n'+title)
        pdf.savefig(fig)
        plt.close()

        # IC, Rank_IC
        fig=plt.figure(figsize=(10, 6))
        result['IC_mean']=np.mean(context.history['IC'])
        result['IC_max']=np.max(context.history['IC'])
        result['IC_min']=np.min(context.history['IC'])
        result['IC_std']=np.std(context.history['IC'], ddof=1)
        result['IC_IR']=np.mean(context.history['IC'])/np.std(context.history['IC'], ddof=1)
        result['IC_T']=result['IC_IR']*np.sqrt(context.history['times']-2)
        result['Rank_IC_mean']=np.mean(context.history['Rank_IC'])
        result['Rank_IC_max']=np.max(context.history['Rank_IC'])
        result['Rank_IC_min']=np.min(context.history['Rank_IC'])
        result['Rank_IC_std']=np.std(context.history['Rank_IC'], ddof=1)
        result['Rank_IC_IR']=np.mean(context.history['Rank_IC'])/np.std(context.history['Rank_IC'], ddof=1)
        result['Rank_IC_T']=result['Rank_IC_IR']*np.sqrt(context.history['times']-2)
        title='IC: 均值:'+'{:.2%}'.format(result['IC_mean'])+\
            '   最大值:'+'{:.2%}'.format(result['IC_max'])+\
            '   最小值:'+'{:.2%}'.format(result['IC_min'])+\
            '   标准差:'+'{:.2%}'.format(result['IC_std'])+\
            '   IR:'+'{:.2f}'.format(result['IC_IR'])+\
            '   T值:'+'{:.2f}'.format(result['IC_T'])+'\n\n'+\
            'Rank_IC: 均值:'+'{:.2%}'.format(result['Rank_IC_mean'])+\
            '   最大值:'+'{:.2%}'.format(result['Rank_IC_max'])+\
            '   最小值:'+'{:.2%}'.format(result['Rank_IC_min'])+\
            '   标准差:'+'{:.2%}'.format(result['Rank_IC_std'])+\
            '   IR:'+'{:.2f}'.format(result['Rank_IC_IR'])+\
            '   T值:'+'{:.2f}'.format(result['Rank_IC_T'])
        x_label=[t.strftime('%Y-%m-%d') for t in context.history['td'][:-1]]

        plt.bar(x_label,context.history['Rank_IC'])
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
