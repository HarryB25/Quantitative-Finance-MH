# 进行投资组合优化，找到最佳的投资组合权重
import time
from datetime import datetime, timedelta

import cvxpy as cvx
import numpy as np
import pandas as pd

from utils.database import mysql_db


def get_days_before(input_date, days=50):
    # 将输入日期从int格式转换为datetime对象
    input_date_str = str(input_date)
    year = int(input_date_str[:4])
    month = int(input_date_str[4:6])
    day = int(input_date_str[6:])
    input_datetime = datetime(year, month, day)

    # 计算days天前的日期
    delta = timedelta(days=days)
    result_datetime = input_datetime - delta

    # 将结果转换为int格式的YYYYMMDD
    result_int_date = result_datetime.year * 10000 + result_datetime.month * 100 + result_datetime.day
    return result_int_date


# 投资组合优化器
class PortfolioOptimizer():
    def __init__(self, data):
        self.data = data

    # 二次规划求解最小方差组合
    def minvar(self, symbols, date):
        weight = cvx.Variable(len(symbols))
        # 读取date前50天的数据，计算每只股票的平均收益率和标准差
        days_before = get_days_before(date, days=60)
        data = self.data[
            (self.data['td'] < date) & (self.data['td'] >= days_before) & (self.data['codenum'].isin(symbols))].copy()
        # 重置索引
        data.reset_index(inplace=True)
        # 计算每只股票的收益率
        data['return'] = data.groupby('codenum')['close'].pct_change()
        panel = data.pivot_table(index='td', columns='codenum', values='return')
        # 计算每只股票的协方差矩阵
        cov = panel.cov()

        # 目标函数min(0.5 * weight.T * cov * weight)
        # s.t. weight.T * 1 = 1, weight >= 0, weight每个分量不超过2/len(symbols)
        objective = cvx.Minimize(cvx.quad_form(weight, cov))
        constraints = [weight.T * np.ones(len(symbols)) == 1, weight >= 0, weight <= 2 / len(symbols)]
        prob = cvx.Problem(objective, constraints)
        prob.solve()
        print("status:", prob.status)
        print("optimal value:", prob.value)
        print("optimal var:", weight.value)

        return weight.value

    def maxprofit(self, symbols, date):
        weight = cvx.Variable(len(symbols))
        return pd.Series(weight)

    def maxsharpe(self, symbols, date):
        weight = cvx.Variable(len(symbols))
        return pd.Series(weight)


if __name__ == '__main__':
    pd.set_option('display.max_rows', 30000)  # 设置最大行数
    pd.set_option('display.max_columns', 30)  # 设置最大列数
    date = 20190101
    con = mysql_db("MH#123456")
    sql = "select td, codenum, close from market where td > %s" % date
    print("从数据库读取数据")
    start = time.time()
    # data = pd.read_sql(sql, con=con)
    data = pd.read_csv('/Users/huanggm/Desktop/Quant/data/astocks_market.csv')[['td', 'codenum', 'close']]
    # 线性填充每只股票的'close'列缺失值
    data['close'] = data.groupby('codenum')['close'].fillna(method='ffill')
    print("数据读取完成")
    optimizer = PortfolioOptimizer(data)
    weight = optimizer.minvar(
        ['000001.SZ', '000002.SZ', '000004.SZ', '000005.SZ', '000006.SZ', '000007.SZ', '000008.SZ', '000009.SZ'], date)
    print(weight)
    end = time.time()
    print("耗时：", end - start)
