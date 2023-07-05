import pymysql
import pandas as pd

def mysql_db(password):
    # 连接数据库需要一些参数，比如用户名、密码、端口号、数据库名
    connection = pymysql.connect(host='172.31.50.91',
                           port=3306,
                           user='guest',
                           password=password
                           )
    return connection

if __name__ == '__main__':
    symbols = ['000001.SZ', '000002.SZ', '000004.SZ', '000005.SZ', '000006.SZ', '000007.SZ', '000008.SZ', '000009.SZ']
    connection = mysql_db()
    # 打开数据库可能会有风险，所以添加异常捕捉
    try:
        with connection.cursor() as cursor:
            # 准备SQL语句
            sql = "select codenum, open, close from astocks.market where codenum in %s;"
            # 执行SQL语句
            cursor.execute(sql, (symbols,))
            # 执行完SQL语句后的返回结果都是保存在cursor中
            # 所以要从cursor中获取全部数据
            datas = cursor.fetchall()
            # 转化为DataFrame格式
            datas = pd.DataFrame(datas)
            # 为DataFrame设置列名
            datas.columns = ['codenum', 'open', 'close']
            # 输出head和tail
            print(datas.head())
            print(datas.tail())
    except Exception as e:
        print("数据库操作异常：\n", e)
    finally:
        # 不管成功还是失败，都要关闭数据库连接
        cursor.close()
        connection.close()