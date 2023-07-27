# 进行投资组合优化，找到最佳的投资组合权重
import time
from datetime import datetime, timedelta

import cvxpy as cvx
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils.database import mysql_db


def get_days_before(input_date, days):
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
    def __init__(self, data=None, risk_free_rate=0.03, days=None):
        self.data = data
        self.risk_free_rate = risk_free_rate
        self.days = days

    def set_data(self, data):
        self.data = data

    def set_days(self, days):
        self.days = days

    def set_risk_free_rate(self, risk_free_rate):
        self.risk_free_rate = risk_free_rate

    def check(self):
        if self.data is None:
            raise ValueError('data is None, please set data first')
        if self.days is None:
            raise ValueError('days is None, please set days first')

    # 二次规划求解最小方差组合
    def minvar(self, symbols, date, subject_to_weight=True):
        self.check()
        weight = cvx.Variable(len(symbols))
        # 读取date前days天的数据，计算每只股票的平均收益率和标准差
        days_before = get_days_before(date, days=self.days)
        data = self.data[
            (self.data['td'] < date) & (self.data['td'] >= days_before) & (self.data['codenum'].isin(symbols))].copy()

        # 重置索引
        data.reset_index(inplace=True)
        # 计算每只股票的收益率
        data['return'] = data.groupby('codenum')['close'].pct_change()
        data.fillna(0, inplace=True)
        panel = data.pivot_table(index='td', columns='codenum', values='return')
        # 计算每只股票的协方差矩阵
        cov = panel.cov()

        # 目标函数min(0.5 * weight.T * cov * weight)
        # s.t. weight.T * 1 = 1, weight >= 0, 如果subject_to_weight=True, weight <= 2 / len(symbols), 否则不加这个约束
        objective = cvx.Minimize(cvx.quad_form(weight, cov))
        # 是否加入权重约束
        if subject_to_weight:
            constraints = [weight.T @ np.ones(len(symbols)) == 1, weight >= 0, weight <= 2 / len(symbols)]
        else:
            constraints = [weight.T @ np.ones(len(symbols)) == 1, weight >= 0]

        # 求解问题
        prob = cvx.Problem(objective, constraints)
        prob.solve()

        return weight.value

    # 求解最大夏普比例组合
    def maxsharpe(self, symbols, date, subject_to_weight=True):
        self.check()
        # 读取date前days天的数据，计算每只股票的平均收益率和标准差
        days_before = get_days_before(date, days=self.days)
        data = self.data[
            (self.data['td'] < date) & (self.data['td'] >= days_before) & (self.data['codenum'].isin(symbols))].copy()
        # 重置索引
        data.reset_index(inplace=True)
        # 计算每只股票的收益率
        data['return'] = data.groupby('codenum')['close'].pct_change()
        data.fillna(0, inplace=True)
        panel = data.pivot_table(index='td', columns='codenum', values='return')
        # 用weight和前self.days天的收益率加权平均计算每只股票的收益率total_return
        total_return = panel.sum(axis=0) / self.days
        total_return = np.array(total_return)

        # 计算每只股票的协方差矩阵
        cov = panel.cov()

        # 目标函数max((weight.T * total_return - free_risk_rate) / sqrt(weight.T * cov * weight))
        # s.t. weight.T * 1 = 1, weight >= 0, 如果subject_to_weight=True, weight <= 2 / len(symbols), 否则不加这个约束

        # 定义目标函数（负夏普比率，因为minimize函数是用于最小化）
        def negative_sharpe_ratio(weights):
            expected_return = total_return.T @ weights
            portfolio_std = np.sqrt(weights.T @ cov @ weights)
            return -(expected_return - self.risk_free_rate) / portfolio_std

        # 设置约束条件（权重之和为1，权重非负，如果subject_to_weight=True, 权重不超过2 / len(symbols)，否则不加这个约束）
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'ineq', 'fun': lambda x: x})
        if subject_to_weight:
            constraints = constraints + ({'type': 'ineq', 'fun': lambda x: 2 / len(symbols) - x},)

        # 设置初始权重值（可以根据实际情况调整）
        initial_weights = np.ones(len(symbols)) / len(symbols)

        # 使用优化算法求解最大化夏普比率的问题
        result = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', constraints=constraints)

        # 获取最优权重向量
        optimal_weights = result.x
        # 将小于1e-4的权重置为0
        optimal_weights[optimal_weights < 1e-4] = 0
        return optimal_weights


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
    optimizer = PortfolioOptimizer()
    optimizer.set_data(data)
    weight = optimizer.minvar(
        ['000001.SZ', '000002.SZ', '000004.SZ', '000005.SZ', '000006.SZ', '000007.SZ', '000008.SZ', '000009.SZ'], date)
    weight_sharpe = optimizer.maxsharpe(
        ['000001.SZ', '000002.SZ', '000004.SZ', '000005.SZ', '000006.SZ', '000007.SZ', '000008.SZ', '000009.SZ'], date,
        subject_to_weight=False)
    end = time.time()
    print("耗时：", end - start)
