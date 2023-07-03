import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pymysql
import numpy as np
from datetime import datetime, timedelta


def generate_date_list(start_date, end_date):
    date_format = "%Y%m%d"  # 日期格式
    date_list = []  # 存储日期的列表

    start = datetime.strptime(start_date, date_format)  # 将起始日期转换为datetime对象
    end = datetime.strptime(end_date, date_format)  # 将结束日期转换为datetime对象

    current_date = start  # 初始化当前日期为起始日期

    # 生成日期列表
    while current_date <= end:
        date_list.append(int(current_date.strftime(date_format)))  # 将当前日期转换为整数并添加到列表中
        current_date += timedelta(days=1)  # 递增一天

    return date_list


def mysql_db():
    # 连接数据库参数
    connection = pymysql.connect(host='172.31.50.91',
                                 port=3306,
                                 user='guest',
                                 password="MH#123456"
                                 )
    return connection


class Account:
    def __init__(self, initial_capital, symbols):
        """
        :param initial_capital:初始资金
        :param symbols:股票范围
        :param long_window:长期均线时间范围
        """
        self.initial_capital = initial_capital  # 初始资金
        self.current_balance = initial_capital  # 当前余额
        self.portfolio_value = initial_capital  # 当前投资组合价值
        self.current_return = 0.0  # 当前收益率
        self.volatility = 0.0  # 波动率
        self.positions = {}  # 持仓
        self.value_history = []  # 投资组合价值历史
        self.trade_history = []  # 交易记录
        for symbol in symbols:
            self.positions[symbol] = 0  # 初始化持仓

    def update_balance(self, amount):
        self.current_balance += amount

    def update_portfolio_value(self, prices):
        self.portfolio_value = self.current_balance
        for symbol in symbols:
            price = prices['close'].loc[prices['symbol'] == symbol].values[-1]
            self.portfolio_value += self.positions[symbol] * price

    def update_return(self):
        self.current_return = (self.portfolio_value - self.initial_capital) / self.initial_capital

    def update_volatility(self):
        self.volatility = np.std(self.value_history)

    def update_position(self, symbol, quantity):
        self.positions[symbol] += quantity

    def record_trade(self, time, symbol, bs, quantity, price):
        trade = {'time': time, 'symbol': symbol, 'buy/sell': bs, 'quantity': quantity, 'price': price}
        self.trade_history.append(trade)


class Strategy:
    def __init__(self, short_window, long_window):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signal(self, group):
        # 计算短期均线和长期均线
        short_ma = group['close'].rolling(self.short_window).mean()
        long_ma = group['close'].rolling(self.long_window).mean()
        signal = 0

        # 金叉信号：短期均线上穿长期均线
        if short_ma.iloc[-1] >= long_ma.iloc[-1] and short_ma.iloc[-2] < long_ma.iloc[-2]:
            signal = 1

        # 死叉信号：短期均线下穿长期均线
        elif short_ma.iloc[-1] <= long_ma.iloc[-1] and short_ma.iloc[-2] > long_ma.iloc[-2]:
            signal = -1
        # 返回交易信号
        return signal


class TradingPlatform:
    def __init__(self, account, strategy):
        self.account = account
        self.strategy = strategy

    # 回测函数
    def run_backtest(self, data):
        # 获取日期列表
        dates = data['td'].unique()
        data1 = data
        dates = list(dates)
        dates.sort()
        # 将数据按照symbol分组
        data = data.groupby('symbol')

        for i in range(0, long_window+1):
            account.value_history.append(initial_capital)
        for i, date in enumerate(dates):
            # 从第long_window天开始回测，因为前long_window天的数据用于计算均线
            if i <= self.strategy.long_window:
                continue
            for symbol, group in data:
                # 获取交易信号和当日收盘价
                signal = self.strategy.generate_signal(group.iloc[:i])
                price = group['close'].loc[group['td'] == date].values[0]  # 获取当日收盘价
                position = self.account.positions # 获取当前持仓数量

                # 生成买卖信号并执行交易
                if signal == 1:
                    # 买入信号且当前未持有该股票
                    available_funds = min(self.account.current_balance,
                                          self.account.portfolio_value * 0.2)  # 可用资金，取当前资金和总资产的20%的较小值
                    if available_funds >= price:
                        quantity = int(available_funds / price)  # 最大可买数量，购买的金额除以当前价格并向下取整
                        self.account.update_balance(-quantity * price)  # 更新账户余额
                        self.account.update_position(symbol, quantity)  # 更新持仓
                        self.account.record_trade(date, symbol, 'buy', quantity, price)  # 记录交易
                        # 输出购买信息
                        print("买入信号：", date, symbol, quantity, price)
                elif signal == -1:
                    # 卖出信号且当前持有该股票
                    quantity = position[symbol]  # 卖出数量
                    self.account.update_balance(quantity * price)
                    self.account.update_position(symbol, -quantity)
                    self.account.record_trade(date, symbol, 'sell', quantity, price)
                    # 输出卖出信息
                    print("卖出信号：", date, symbol, quantity, price)

            prices = data1.loc[data1['td'] == date, ['symbol', 'close']]
            print(prices)
            # 更新投资组合价值
            self.account.update_portfolio_value(prices)
            print("投资组合价值：", self.account.portfolio_value)
            # 输出持仓
            print("持仓：", self.account.positions)
            print()
            # 更新账户信息
            self.account.update_return()
            self.account.value_history.append(self.account.portfolio_value)
            self.account.update_volatility()

        print("回测完成！")

    def plot_portfolio_value(self, data, data_500):
        dates = data['td'].unique()
        dates = list(dates)
        dates.sort()
        values = account.value_history
        values_500 = data_500['close'].loc[data_500['td'].isin(dates)].values
        values_500 = values_500 / values_500[0] * account.initial_capital
        date_values_dict = dict(zip(dates, values))
        # 获取交易历史
        trades = account.trade_history
        buy_time_list = []
        buy_value_list = []
        sell_time_list = []
        sell_value_list = []
        for i, trade in enumerate(trades):
            if trade['buy/sell'] == 'buy':
                buy_time_list.append(trade['time'])
                buy_value_list.append(date_values_dict[trade['time']])
            else:
                sell_time_list.append(trade['time'])
                sell_value_list.append(date_values_dict[trade['time']])
        # 将日期字符串转换为datetime对象
        date_objects = [datetime.strptime(str(date), "%Y%m%d") for date in dates]
        buy_date_objects = [datetime.strptime(str(date), "%Y%m%d") for date in buy_time_list]
        sell_date_objects = [datetime.strptime(str(date), "%Y%m%d") for date in sell_time_list]
        # 创建图表和子图
        fig, ax = plt.subplots(figsize=(18, 6))
        # 绘制图表
        ax.plot(date_objects, values, label='投资组合', color='blue')
        ax.plot(date_objects, values_500, label='中证500', color='red')
        # 设置x轴刻度定位器为年份
        ax.xaxis.set_major_locator(mdates.YearLocator())
        # 设置x轴刻度格式为YYYYMMDD
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y%m%d"))
        # 自动调整x轴标签的显示以避免重叠
        fig.autofmt_xdate()
        ax.scatter(buy_date_objects, buy_value_list, marker='^', color='green', label='Buy')
        ax.scatter(sell_date_objects, sell_value_list, marker='v', color='red', label='Sell')
        plt.title('Portfolio Value')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.grid(True)
        plt.show()

    def plot_trades(self, data):
        dates = data.index
        portfolio_values = data['portfolio_value']
        buy_signals = data[data['signal'] == 1]
        sell_signals = data[data['signal'] == -1]

        plt.figure(figsize=(12, 6))
        plt.plot(dates, portfolio_values, label='Portfolio Value')
        plt.scatter(buy_signals.index, buy_signals['portfolio_value'], marker='^', color='green', label='Buy')
        plt.scatter(sell_signals.index, sell_signals['portfolio_value'], marker='v', color='red', label='Sell')
        plt.title('Portfolio Value with Buy/Sell Signals')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_trade_history(self, file_path):
        df = pd.DataFrame(self.account.trade_history)
        df.to_csv(file_path, index=False)


# 建立数据库连接
connection = mysql_db()
# 交易的股票范围
symbols = ['000011.SZ', '000012.SZ', '000014.SZ', '000015.SZ', '000016.SZ', '000017.SZ', '000018.SZ', '000019.SZ']
# 回测的时间范围
start_date = 20200101
end_date = 20221231

cursor = connection.cursor()
# 准备SQL语句
sql = "select td, codenum,open,close from astocks.market where codenum in %s and td between " + str(
    start_date) + " and " + str(end_date) + " order by codenum asc;"
# 执行SQL语句
cursor.execute(sql, (symbols,))
# 执行完SQL语句后的返回结果都是保存在cursor中
# 所以要从cursor中获取全部数据
datas = cursor.fetchall()
# 转化为DataFrame格式
datas = pd.DataFrame(datas)
# 为DataFrame设置列名
datas.columns = ['td', 'symbol', 'open', 'close']

sql_500 = "select td, close from astocks.indexprice where indexnum = '000905.SH' and td between " + str(
    start_date) + " and " + str(end_date) + " order by td asc;"
cursor.execute(sql_500)
datas_500 = cursor.fetchall()
datas_500 = pd.DataFrame(datas_500)
datas_500.columns = ['td', 'close']

# 对空缺值进行填充
datas = datas.fillna(method='ffill')

initial_capital = 100000
short_window = 3
long_window = 5
account = Account(initial_capital, symbols)
strategy = Strategy(short_window=short_window, long_window=long_window)
platform = TradingPlatform(account, strategy)
platform.run_backtest(datas)
print(account.portfolio_value)

# 绘制投资组合价值增长过程
platform.plot_portfolio_value(datas, datas_500)

# 保存历史交易记录为CSV文件
platform.save_trade_history('trade_history.csv')