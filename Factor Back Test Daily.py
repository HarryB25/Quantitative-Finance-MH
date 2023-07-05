from utils.database import mysql_db
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import re
from itertools import product
import calendar
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy.stats import spearmanr


# 获取从start_date到end_date之间每个季度的最后一天，即3、6、9、12月的最后一天
def get_quarter_list(start_date, end_date):
    # 将输入的日期字符串转换为datetime对象
    start_date = str(start_date)
    end_date = str(end_date)
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')

    month_list = []

    # 从start_dt到end_dt逐月迭代
    current_dt = start_dt
    while current_dt <= end_dt:
        month = current_dt.month

        # 检查是否是季度的最后一个月（3、6、9、12月）
        if month in [3, 6, 9, 12]:
            # 获取该月份的最后一天
            last_day = calendar.monthrange(current_dt.year, month)[1]

            # 创建季度最后一天的日期对象，并添加到month_list中
            quarter_last_day = datetime(current_dt.year, month, last_day)
            month_list.append(int(quarter_last_day.strftime('%Y%m%d')))

        # 增加一个月
        current_dt = current_dt.replace(day=1)
        current_dt = current_dt + relativedelta(months=1)

    return month_list


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


def RankIC(df1, df2):
    # df1, df2为两个DataFrame，index为股票代码，columns为时间
    # 计算每个时间点的IC值
    IC = []
    for time in df1.columns:
        df = pd.concat([df1[time], df2[time]], axis=1)
        df.columns = ['factor', 'return']
        df = df.dropna()
        IC.append(spearmanr(df['factor'], df['return'])[0])
    return IC


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
            price = prices.loc[prices['codenum'] == symbol, 'close'].values[0]
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
            self.dates = date_list
        else:
            self.dates = get_quarter_list(start_date, end_date)

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
                 weights='equal'
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
        self.turnover_dates = get_quarter_list(start_date, end_date)  # 获取每个季度的最后一天的日期
        self.weights = weights

    def run(self):
        # 获取数据并更新股票池
        original_datas = pd.read_csv('/Users/huanggm/Desktop/Quant/data/astocks_market_deriv.csv')
        # 取出td, codenum, PB三列
        original_datas = original_datas[['td', 'codenum', self.factor_name]]
        original_prices = pd.read_csv('/Users/huanggm/Desktop/Quant/data/astocks_market.csv')
        original_prices = original_prices[['td', 'codenum', 'close']]
        original_datas = original_datas.merge(original_prices, on=['td', 'codenum'])  # 合并数据
        original_datas['ROE'] = 1 / original_datas[self.factor_name]
        self.turnover_dates = list(original_datas['td'].unique())  # 获取每个季度的最后一天的日期
        for date in self.turnover_dates:
            print('回测日期：', date)
            # 获取数据并更新股票池
            datas = original_datas[original_datas['td'] == date]  # 获取当前日期的数据
            datas = datas.sort_values(['ROE', 'codenum'], ascending=False)  # 按照因子大小和股票代码排序,降序
            datas = datas.reset_index(drop=True)  # 重置索引

            datas = datas.head(int(len(self.pool.symbols)*0.1))  # 取前10%股票
            self.pool.symbols = datas['codenum'].tolist()  # 获取当前日期的股票代码列表
            prices = original_prices[original_prices['td'] == date]  # 获取当前日期的股票价格
            dates = list(prices['td'].unique())  # 获取当前日期的股票价格的日期列表
            symbols = self.pool.symbols  # 获取股票池中的股票
            self.account.symbols = symbols  # 更新account的股票池
            self.pool.symbols = sorted(self.pool.symbols)  # 对股票池中的股票代码进行排序
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

            datas = datas.merge(prices, on=['td', 'codenum'], how='left')  # 合并datas和prices
            # 合并datas和weights
            datas = datas.merge(weights, left_on='codenum', right_index=True, how='left')
            print(datas)
            datas['shares'] = datas['weights'] * self.account.current_balance / datas['close']  # 按照权重计算每只股票的持仓数量
            self.account.current_balance -= np.sum(datas['shares'] * datas['close'])  # 调整账户余额
            # 保存持仓数量到account的持仓数量字典中
            for symbol in symbols:
                self.account.update_position(symbol, datas.loc[datas['codenum'] == symbol, 'shares'].values[0])
            print('日期：', date)
            prices_sub = prices
            self.account.update_portfolio_value(prices_sub, date)
            print("投资组合价值：", self.account.portfolio_value)
            # 输出持仓
            print("持仓：", self.account.positions)
            self.account.value_history.append([date, self.account.portfolio_value])

            # 更新账户股票池
            self.account.sell_all(prices)

        self.account.update_return()
        self.account.update_volatility()
        # 将self.account.value_history保存为csv文件
        value_history = pd.DataFrame(self.account.value_history, columns=['date', 'portfolio_value'])
        value_history.to_csv('portfolio_value.csv', index=False)
        self.plot()

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

        # 找出小于max_drawdown_idx的最大的peak
        previous_peaks = [i for i in peaks if i < max_drawdown_idx]

        # 找到最大回撤的起始点和结束点
        start_idx = drawdown.idxmax()
        end_idx = previous_peaks[-1]

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
                     xytext=(date_objects[end_idx - 23], portfolio_value[end_idx]),
                     arrowprops=dict(facecolor='red', arrowstyle='->'))

        plt.show()


pd.set_option('display.max_rows', 30000)  # 设置最大行数
pd.set_option('display.max_columns', 30)  # 设置最大列数

backtest = BackTest(factor_name='PB', start_date=20211231, end_date=20221231)
backtest.run()
