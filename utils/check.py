import re

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


def check_factor(factor, password):
    # 检查astocks.finance_deriv表中是否有名为factor的属性列
    connection = mysql_db(password)  # 连接数据库
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
