from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import ttest_1samp


def RankIC(factor, price):
    print(factor.head())
    print(price.head())
    # 保留td和codenum、close列
    price = price[['td', 'codenum', 'close']]
    # 按照日期和股票代码排序
    price = price.sort_values(['td', 'codenum'])
    # 计算每只股票的日收益率rank
    price['return'] = price.groupby('codenum')['close'].pct_change()
    # 将return在日期上提前一天
    price['return'] = price.groupby('codenum')['return'].shift(-1)
    # 计算每只股票的日收益率rank
    price['return_rank'] = price.groupby('td')['return'].rank()
    # 按照日期和股票代码排序
    factor = factor.sort_values(['td', 'codenum'])
    # 新建列ROE=1/PB
    # factor['ROE'] = 1 / factor['PB']
    # 计算因子rank
    factor['factor_rank'] = factor.groupby('td')['factor'].rank()
    factor = factor[['td', 'codenum', 'factor_rank']]
    # 保留factor左连接合并数据
    data = pd.merge(factor, price, on=['td', 'codenum'], how='left')
    # 计算RankIC
    RankIC = []
    dates = sorted(data['td'].unique())
    df = data.dropna()
    print(dates)
    for date in dates:
        # 计算每日的RankIC
        df_tmp = df[df['td'] == date]
        corr = spearmanr(df_tmp['return_rank'], df_tmp['factor_rank'])[0]
        print('date:', date, 'corr:', corr)
        RankIC.append([date, corr])
    IC = RankIC
    IC = IC[:-1]
    # 去掉空值

    date_objects = [datetime.strptime(str(date[0]), "%Y%m%d") for date in IC]

    # 设置画幅比例
    plt.figure(figsize=(20, 6))
    # 绘图
    plt.plot(date_objects, [i[1] for i in IC])

    # 计算IC均值
    IC_mean = np.mean([i[1] for i in IC])
    # 计算IC标准差
    IC_std = np.std([i[1] for i in IC])
    # 计算IC t值
    t = IC_mean / (IC_std / np.sqrt(len(IC)))
    # 计算IC p值
    p = ttest_1samp([i[1] for i in IC], 0)[1]
    # 输出结果
    print('RankIC均值：', IC_mean)
    print('RankIC标准差：', IC_std)
    print('RankIC t值：', t)
    print('RankIC p值：', p)

    # 计算ICIR
    ICIR = IC_mean / IC_std
    # 输出结果
    print('ICIR：', ICIR)

    plt.title('RankIC')
    plt.xlabel('date')
    plt.ylabel('RankIC')
    plt.show()
