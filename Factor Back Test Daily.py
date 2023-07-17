import re
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

from utils.database import mysql_db


def is_valid_date(date):
    date_str = str(date)

    # 检查日期长度是否为8
    assert len(date_str) == 8, "Invalid date format. Should be YYYYMMDD."

    # 检查日期是否为YYYMMDD格式
    assert re.match(r'^[0-9]{4}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])$',
                    date_str), "Invalid date format. Should be YYYYMMDD."

    # 提取年月日
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:])

    # 检查年月日是否合法
    assert year > 0, "Invalid year."
    assert 1 <= month <= 12, "Invalid month."
    assert 1 <= day <= 31, "Invalid day."

    # 检查特殊月份日期是否合法
    if month in [4, 6, 9, 11]:
        assert day <= 30, "Invalid day for the given month."
    elif month == 2:
        if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
            assert day <= 29, "Invalid day for the given month."
        else:
            assert day <= 28, "Invalid day for the given month."

    return True


def check_factor(factor):
    # 检查astocks.finance_deriv表中是否有名为factor的属性列
    connection = mysql_db()  # 连接数据库
    sql = f"SELECT {factor} FROM astocks.market_deriv WHERE ;"
    try:
        cursor = connection.cursor()
        cursor.execute(sql)
        datas = cursor.fetchall()
        if len(datas) == 0:
            raise Exception(f"astocks.finance_deriv表中没有名为{factor}的属性列")
    except Exception as e:
        print(e)
    finally:
        connection.close()

class Account:
    def __init__(self, initial_capital, symbols=None):
        """
        :param initial_capital:初始资金
        :param symbols:股票范围
        """
        self.initial_capital = initial_capital  # 初始资金
        self.current_balance = initial_capital  # 当前余额
        self.portfolio_value = initial_capital  # 当前投资组合价值
        self.current_return = 0.0  # 当前收益率
        self.volatility = 0.0  # 波动率
        self.positions = {}  # 持仓
        self.value_history = []  # 投资组合价值历史
        self.trade_history = []  # 交易记录
        self.symbols = symbols  # 股票范围
        for symbol in symbols:
            self.positions[symbol] = 0  # 初始化持仓

    def update_balance(self, amount):
        self.current_balance += amount

    def update_portfolio_value(self, prices, date):
        self.portfolio_value = self.current_balance
        for symbol in self.symbols:
            price = prices.loc[prices['codenum'] == symbol, 'close'].values[0]
            self.portfolio_value += self.positions[symbol] * price
        self.value_history.append([date, self.portfolio_value])

    def sell_all(self, prices):
        for symbol in self.symbols:
            # 判断prices中是否有symbol的数据
            data = prices.loc[prices['codenum'] == symbol, 'close']
            if len(data) == 0:
                continue
            else:
                price = data.values[0]
                self.current_balance += self.positions[symbol] * price
                self.positions[symbol] = 0
        self.positions = {}

    def update_return(self):
        self.current_return = (self.portfolio_value - self.initial_capital) / self.initial_capital

    def update_volatility(self):
        self.volatility = np.std([x[1] for x in self.value_history])

    def update_position(self, symbol, quantity):
        self.positions[symbol] += quantity

    def record_trade(self, time, symbol, bs, quantity, price):
        trade = {'time': time, 'symbol': symbol, 'buy/sell': bs, 'quantity': quantity, 'price': price}
        self.trade_history.append(trade)


# 股票池类
class StockPool:
    def __init__(self, start_date, end_date, symbols=None):

        self.symbols = symbols  # 股票代码
        self.start_date = start_date  # 日期
        self.end_date = end_date
        # 如果symbols为空，则默认为沪深300
        if symbols is None:
            df = pd.read_csv('/Users/huanggm/Desktop/Quant/data/astocks_indexweight.csv')
            df = df.loc[(df['td'] >= start_date) & (df['td'] <= end_date) & (df['indexnum'] == '000300.SH')]
            date_list = list(df['td'].unique())
            self.symbols = list(df.loc[df['td'] == date_list[0], 'code'])

    def get_datas(self, factor_name, start_date, end_date):
        """
        :param factor_name: 因子名称
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return: 返回数据
        """
        # 从数据库中获取股票数据
        check_factor(factor_name)
        connection = mysql_db()
        sql = f"SELECT td, codenum, {factor_name} " \
              f"FROM astocks.market_deriv " \
              f"WHERE codenum IN %s AND td between {start_date} AND {end_date};"
        cursor = connection.cursor()  # 建立游标
        cursor.execute(sql, (self.symbols,))  # 执行sql语句
        datas = cursor.fetchall()  # 获取查询结果
        datas = pd.DataFrame(datas)  # 转换为DataFrame格式
        datas.columns = ['td', 'codenum', 'factor']  # 重命名列名

        datas = datas.dropna(axis=0, how='any')  # 删除空值，不过感觉没必要，因为在数据库连接时已经删除了空值
        self.datas = datas  # 保存数据
        connection.close()  # 关闭数据库连接
        return self.datas

    def get_price(self, start_date, end_date):
        connection = mysql_db()
        sql = f"SELECT td, codenum, close FROM astocks.market WHERE codenum IN %s AND td BETWEEN {start_date} AND {end_date};"
        cursor = connection.cursor()
        cursor.execute(sql, (self.symbols,))
        datas = cursor.fetchall()
        datas = pd.DataFrame(datas)
        datas.columns = ['td', 'codenum', 'close']
        connection.close()
        # 线性填充空值
        datas['close'] = datas.groupby('codenum')['close'].apply(lambda x: x.fillna(method='ffill'))
        return datas


# 回测类
class BackTest:
    def __init__(self, factor_name, initial_capital=100000, symbols=None,
                 start_date=int(time.strftime('%Y%m%d', time.localtime(time.time() - 365 * 24 * 60 * 60))),
                 end_date=int(time.strftime('%Y%m%d', time.localtime(time.time()))),
                 weights='equal', drawdown_num=5
                 ):
        """
        :param factor_name: 因子名称
        :param initial_capital: 初始资金
        :param symbols: 股票代码列表
        :param start_date: 起始日期
        :param end_date: 结束日期
        :param weights: 权重方式
        """

        # 参数合法性检查
        is_valid_date(start_date)
        is_valid_date(end_date)
        assert isinstance(factor_name, str), "factor_name必须为字符串"
        assert weights in ('random', 'equal'), "权重参数为'random'或'equal'"
        if symbols is not None and len(symbols) > 0:
            assert isinstance(symbols, list), "symbols必须为列表"

        # 初始化
        self.start_date = start_date
        self.end_date = end_date
        self.pool = StockPool(symbols=symbols, start_date=start_date, end_date=end_date)
        self.symbols = symbols
        symbols = self.pool.symbols
        self.account = Account(initial_capital=initial_capital, symbols=symbols)
        self.factor_name = factor_name
        self.weights = weights
        self.weights = weights
        self.drawdown_num = drawdown_num
        self.turnover_dates = None

    def run(self):
        # 获取数据并更新股票池
        original_datas = pd.read_csv('/Users/huanggm/Desktop/Quant/data/astocks_market_deriv.csv')
        # 取出td, codenum, PB三列
        original_datas = original_datas[
            (original_datas['td'] >= self.start_date) & (original_datas['td'] <= self.end_date)]
        original_datas = original_datas[['td', 'codenum', self.factor_name]]
        original_datas[f'{self.factor_name}'] = 1 / original_datas[f'{self.factor_name}']
        original_prices = pd.read_csv('/Users/huanggm/Desktop/Quant/data/astocks_market.csv')
        original_prices = original_prices[
            (original_prices['td'] >= self.start_date) & (original_prices['td'] <= self.end_date)]
        original_prices = original_prices[['td', 'codenum', 'close']]
        original_datas = original_datas.merge(original_prices, on=['td', 'codenum'])  # 合并数据
        # original_datas['ROE'] = 1 / original_datas[self.factor_name]
        self.turnover_dates = list(original_datas['td'].unique())  # 获取每个季度的最后一天的日期

        for i, date in enumerate(self.turnover_dates):
            print('回测日期：', date)
            # 获取昨日股票池的今日价格
            prices = original_datas[original_datas['td'] == date][['codenum', 'close']]
            if i > 0:
                self.account.sell_all(prices)  # 卖出所有股票
            datas = original_datas[original_datas['td'] == date]  # 获取当前日期的数据
            symbols = datas['codenum'].tolist()  # 获取当前日期的股票代码列表
            datas = datas.sort_values([f'{self.factor_name}', 'codenum'], ascending=False)  # 按照因子大小和股票代码排序,降序
            datas = datas.reset_index(drop=True)  # 重置索引
            datas = datas.head(int(len(symbols) * 0.05))  # 取前10%股票
            symbols = datas['codenum'].tolist()  # 获取当前日期的股票代码列表
            self.pool.symbols = symbols  # 获取当前日期的股票代码列表
            self.account.symbols = symbols  # 更新account的股票池
            print('股票池：', self.pool.symbols)

            # 初始化account的持仓数量字典
            for symbol in symbols:
                self.account.positions[symbol] = 0

            # 如果weights为random，则随机生成权重,否则默认为等权重
            if self.weights == 'random':
                weights = np.random.random(len(self.pool.symbols))
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(self.pool.symbols)) / (len(self.pool.symbols))  # 默认权重为等权重
            weights = pd.DataFrame(weights, index=symbols)  # 转换为DataFrame格式
            weights.columns = ['weights']
            # 合并datas和weights
            datas = datas.merge(weights, left_on='codenum', right_index=True, how='left')
            datas['shares'] = datas['weights'] * self.account.current_balance / datas['close']  # 按照权重计算每只股票的持仓数量
            self.account.current_balance -= np.sum(datas['shares'] * datas['close'])  # 调整账户余额

            # 保存持仓数量到account的持仓数量字典中
            for symbol in symbols:
                # print('股票代码：', symbol)
                self.account.update_position(symbol, datas.loc[datas['codenum'] == symbol, 'shares'].values[0])

            print('日期：', date)
            prices_sub = prices
            self.account.update_portfolio_value(prices_sub, date)
            print("投资组合价值：", self.account.portfolio_value)

            # 输出持仓
            print("持仓：", self.account.positions)
            self.account.value_history.append([date, self.account.portfolio_value])

            # 更新账户股票池
            # self.account.sell_all(prices)

        self.account.update_return()  # 更新收益率
        self.account.update_volatility()  # 更新波动率

        # 将self.account.value_history保存为csv文件
        value_history = pd.DataFrame(self.account.value_history, columns=['date', 'portfolio_value'])
        value_history.to_csv('portfolio_value.csv', index=False)

        # 画图
        self.plot()
        self.plot_return()
        self.RankIC(original_datas, original_prices, self.factor_name)

    # 定义绘图函数
    def plot(self):
        plt.figure(figsize=(15, 5))
        self.account.value_history = sorted(self.account.value_history, key=lambda x: x[0])
        date_objects = [datetime.strptime(str(date[0]), "%Y%m%d") for date in self.account.value_history]
        portfolio_value = pd.Series([i[1] for i in self.account.value_history])
        # 计算每次回撤
        previous_peak = portfolio_value.cummax()
        peaks = portfolio_value[portfolio_value == previous_peak].index.tolist()

        drawdown = (previous_peak - portfolio_value) / previous_peak
        max_drawdown = drawdown.max()
        max_drawdown_idx = drawdown.idxmax()

        # 计算回撤个数
        drawdown_count = len(previous_peak)
        if drawdown_count < self.drawdown_num:
            drawdown_num = drawdown_count - 1
        else:
            drawdown_num = self.drawdown_num

        # 找到每一次从一个peak到突破这个peak的范围
        drawdown_range = []
        for i in range(len(peaks) - 1):
            for j in range(peaks[i], peaks[i + 1]):
                if portfolio_value[j] > portfolio_value[peaks[i]]:
                    drawdown_range.append([peaks[i], j])
                    break
        # 添加最后一个peak到最后一个点的范围
        drawdown_range.append([peaks[-1], len(portfolio_value) - 1])

        # 找到最长的drawdown_num个范围
        drawdown_range = sorted(drawdown_range, key=lambda x: x[1] - x[0], reverse=True)
        drawdown_range = drawdown_range[:drawdown_num]
        print(drawdown_range)
        # 找出小于max_drawdown_idx的最大的peak
        previous_peaks = [i for i in peaks if i < max_drawdown_idx]

        # 找到最大回撤的起始点和结束点
        start_idx = drawdown.idxmax()
        end_idx = previous_peaks[-1]

        # 绘制最大的drawdown_num个回撤，使其背景为灰色

        # 绘制价格序列图
        plt.figure(figsize=(15, 5))
        plt.plot(date_objects, portfolio_value)
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.title('Portfolio Value Series')

        # 绘制回撤的箭头
        arrow_y = portfolio_value[start_idx]
        arrow_text = f'{max_drawdown:.2%}'
        plt.annotate(arrow_text, xy=(date_objects[start_idx], arrow_y),
                     xytext=(date_objects[start_idx + 1], arrow_y),
                     arrowprops=dict(facecolor='red', arrowstyle='->'))
        # 在end_idx处标注一个绿色的点
        plt.plot(date_objects[end_idx], portfolio_value[end_idx], 'o', color='g')
        # 在start_idx处标注一个绿色的点
        plt.plot(date_objects[start_idx], portfolio_value[start_idx], 'o', color='g')
        # 在每个drawdown_range[0]和drawdown_range[1]之间的区域画一个灰色的背景
        for i in range(drawdown_num):
            plt.axvspan(drawdown_range[i][0], drawdown_range[i][1], facecolor='gray', alpha=0.5)
        plt.show()
        '''
        portfolio_value = pd.Series([i[1] for i in self.account.value_history], index=date_objects)
        portfolio_value.index = pd.to_datetime(portfolio_value.index)
        print(portfolio_value.head())
        pf.create_returns_tear_sheet(portfolio_value, benchmark_rets=None, return_fig=True)
        pf.plot_drawdown_periods(portfolio_value, top=5)
        '''

    # 定义每日收益柱状图绘制函数
    def plot_return(self):
        return_history = []
        plt.figure(figsize=(15, 5))
        # 计算每日收益，用value_history的当前值减去前一天的值
        for i in range(1, len(self.account.value_history)):
            return_history.append(
                [self.account.value_history[i][0],
                 self.account.value_history[i][1] - self.account.value_history[i - 1][
                     1]])
        return_history = pd.DataFrame(return_history, columns=['date', 'return'])
        return_history['date'] = pd.to_datetime(return_history['date'], format='%Y%m%d')
        # 绘制每日收益柱状图,红色为正收益，绿色为负收益
        plt.bar(return_history['date'][return_history['return'] >= 0],
                return_history['return'][return_history['return'] >= 0], color='red')
        plt.bar(return_history['date'][return_history['return'] < 0],
                return_history['return'][return_history['return'] < 0], color='green')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.title('Daily Return')
        plt.show()

    def RankIC(self, factor, price, factor_name):
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
        factor['factor_rank'] = factor.groupby('td')[factor_name].rank()
        factor = factor[['td', 'codenum', 'factor_rank']]
        # 保留factor左连接合并数据
        data = pd.merge(factor, price, on=['td', 'codenum'], how='left')
        # 计算RankIC
        RankIC = []
        dates = sorted(data['td'].unique())
        df = data.dropna()
        for date in dates:
            # 计算每日的RankIC
            df_tmp = df[df['td'] == date]
            corr = spearmanr(df_tmp['return_rank'], df_tmp['factor_rank'])[0]
            # print('date:', date, 'corr:', corr)
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
        # 计算ICIR
        ICIR = IC_mean / IC_std

        # 输出结果
        print('RankIC均值：', IC_mean)
        print('RankIC标准差：', IC_std)
        print('RankIC t值：', t)
        print('RankIC p值：', p)
        print('ICIR：', ICIR)

        plt.title('RankIC')
        plt.xlabel('date')
        plt.ylabel('RankIC')
        plt.show()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 30000)  # 设置最大行数
    pd.set_option('display.max_columns', 30)  # 设置最大列数

    backtest = BackTest(factor_name='PB', start_date=20220331, end_date=20221231, drawdown_num=3)
    backtest.run()
