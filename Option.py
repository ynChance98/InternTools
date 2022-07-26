#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pathlib import Path
import pandas as pd
import bsm
import calendar
import datetime as dt
import warnings

warnings.filterwarnings('ignore')


 

class File(object):
    """
    获取文件路径
    """
    def __init__(self, category, directory, tradingm):
        self.category = category
        self.directory = directory
        self.tradingm = str(tradingm)
        self.path = self.get_file()

    def get_file(self):
        if self.category == '50etf':
            if self.tradingm < '201903':
                path_list = [path for path in                              Path(self.directory).joinpath(self.category).joinpath(self.tradingm[:4])                                  .joinpath(self.tradingm).joinpath(self.tradingm).iterdir() if path.is_file()]
            else:
                path_list = [path for path in                              Path(self.directory).joinpath(self.category).joinpath(self.tradingm[:4])                                  .joinpath(self.tradingm).iterdir() if path.is_file()]
        else:
            path_list = [path for path in                          Path(self.directory).joinpath(self.category)                              .joinpath(self.tradingm).iterdir() if path.is_file()]
        return path_list


class Option(object):
    def __init__(self, category, directory, tradingm, begin_time_list, end_time_list, interval):
        """
        :param category: 期权类别
        :param directory:  数据所在根目录
        :param tradingm: 交易月份
        :param begin_time_list: 交易开始时间列表
        :param end_time_list: 交易结束时间列表
        :param interval: 原始数据最小时间间隔
        """
        self.category = category
        self.directory = directory
        self.tradingm = tradingm
        self.begin_time_list = begin_time_list
        self.end_time_list = end_time_list
        self.interval = interval
        self.path_list = File(category, directory, tradingm).path
        self.data_m = self.concat()
        self.data_m['iv'] = bsm.BsmMatrix(
            self.data_m[['t', 'cp', 'strike', 'forward', 'opt_price', 'r', 'vol']]).calc_iv().iv

    def preprocessing(self, path):
        """
        处理一个文件
        :param path:
        :return:
        """
        if self.tradingm > '202101':
            if self.tradingm > '202103':
                data = pd.read_csv(path, index_col=False)
            else:
                data = pd.read_excel(path, index_col=False)  # excel格式
            data.columns = ['time', 'code', 'tradecode', 'strike', 'open', 'high', 'low',
                            'close', 'volume', 'openinterest', 'amount', 'multiplier',
                            'expirydate', 'spotcode', 'spotclose']  # 列名rename，feature有所变化，进行统一
            data.time = pd.to_datetime(data.time)
            data = data.set_index('time')
            data['time'] = data.index.time.astype(str)
            data.time = data.time.str[:5]
            data['date'] = data.index.date.astype(str)
            if self.category == "300股指":
                data.tradecode = data.code
        else:
            data = pd.read_csv(path, index_col=False)

        column = data.columns.tolist()  # 表头出错
        if self.tradingm < '201909' and column[1] != 'time':
            column.insert(1, 'time')
            column = column[:-1]
            data.columns = column
        if column[-4] == 'time.1':
            column.pop(-4)
            column.append('1')
            data.columns = column
            data = data.drop('1', axis=1)
        if type(data.tradecode[0]) != str:
            column.pop(2)
            column.append('1')
            data.columns = column
            data = data.drop('1', axis=1)
        print(column)

        data = data.drop_duplicates(subset=['time', 'date'], keep='last')  # 数据重复

        process = pd.DataFrame()
        for i in range(len(self.begin_time_list)):  # 提取交易时间内的数据
            cut = data[(data.time >= self.begin_time_list[i]) & (data.time <= self.end_time_list[i])]
            process = process.append(cut)
        process['time'] = process.date + ' ' + process.time
        process['time'] = pd.to_datetime(process.time)
        process = process.set_index('time')
        process = process.sort_index()

        if '201909' <= self.tradingm <= '202101':  # rename
            process = process.rename(columns={'oi': 'openinterest'})
        process = process[
            ['date', 'expirydate', 'code', 'tradecode', 'volume', 'openinterest', 'strike', 'spotclose', 'close']]
        if self.category == '300股指':  # 提取cp
            process['cp'] = process.tradecode.str.split('-').str[1]
        else:
            if self.category == '50etf' and self.tradingm == '202103':
                process['cp'] = process.tradecode.str[5]
            else:
                process['cp'] = process.tradecode.str[6]
        process.cp = process.cp.replace(['C', 'P'], [1, -1])
        process.cp = process.cp.replace(['购', '沽'], [1, -1])

        process = process.rename(columns={'spotclose': 'forward', 'close': 'opt_price'})  # rename

        process['right'] = process.forward.apply(lambda x: type(x) == float)  # 数据错位 forward里出现L1
        process = process[process.right]
        process = process.drop(['right'], axis=1)

        process.expirydate = process.expirydate.astype(str)
        process.date = pd.to_datetime(process.date).astype(str)
        return process

    def init_align(self, tradingday_list):
        align = pd.DataFrame()
        for tradingday in tradingday_list:
            day = init_time_axis(tradingday, self.begin_time_list, end_time_list, self.interval)
            align = align.append(day)
        return align

    def concat(self):
        result = []
        for path in self.path_list:
            try:
                data = self.preprocessing(path)
            except:
                continue
            if data.empty:
                continue
            tradingday_list = data.date.unique().tolist()
            data.volume = data.volume.fillna(0)  # 原始数据中volume缺失
            align = self.init_align(tradingday_list)
            align = pd.concat([align, data], axis=1)

            align['count'] = align.groupby(['tradingday']).expirydate.transform('count')  # 剔除缺失率大于50%的交易日
            align = align[align['count'] > 25]

            if align.empty:
                continue
            align = align.drop(['count'], axis=1)

            volume = align.volume
            align.forward = align.forward.replace(0, None)  # 有些forward是0，需要进行填充
            x = []
            for date in tradingday_list:  # 单日内填充
                cut = align[align.tradingday == date]
                cut = cut.fillna(method='ffill')
                cut = cut.fillna(method='bfill')
                x.append(cut)
            align = pd.concat(x)
            align.volume = volume

            align.expirydate = pd.to_datetime(align.expirydate).astype(str)  # 计算成熟期t
            align.expirydate = align.expirydate.str.replace('-', '')
            align.date = align.index.date.astype(str)
            align.date = align.date.str.replace('-', '')
            if type(align.apply(lambda row: trading_day_diff(row.date, row.expirydate), axis=1) / 365) == pd.DataFrame:
                continue
            else:
                align['t'] = (align.apply(lambda row: trading_day_diff(row.date, row.expirydate), axis=1) / 365)
            align = align[align.t >= 0]

            if self.tradingm >= '201909':  # 降采样为5min
                align['minute'] = align.index.time.astype(str)
                align.minute = align.minute.str[3:5].astype(int)
                align = align[align.minute % 5 == 0]
                align = align.drop(columns=['minute'])
            result.append(align)
        total = pd.concat(result)
        total['count'] = total.groupby(['tradingday', 'expirydate', 'strike']).volume.transform('count')
        total = total[total['count'] > 50]  # 买卖数据条数之和在50条以上
        total.volume = total.volume.fillna(0)
        total['r'] = 0.02
        total['vol'] = 0.5
        total = total.drop(['count', 'tradingday'], axis=1)
        total = total.reset_index(drop=False)
        return total


# month_list = list(range(201502, 201513))
# month_list += list(range(201601, 201613))
# month_list += list(range(201701, 201713))
# month_list += list(range(201801, 201813))
# month_list += list(range(201901, 201913))
# month_list += list(range(202001, 202013))
# month_list += list(range(202101, 202108))

month_list = [201912]
month_list += list(range(202001, 202003))
month_list += list(range(202101, 202108))

category = '深300etf'
begin_time_list = ['09:30', '13:00']
end_time_list = ['11:30', '15:00']
directory = 'D:/VSCODE/东莞证券/期权材料/'
#for i in month_list:
#    tradingm = str(i)
#    if tradingm < '201909':
#        data = Option(category, directory, tradingm, begin_time_list, end_time_list, '5T').data_m
#    else:
#        data = Option(category, directory, tradingm, begin_time_list, end_time_list, '1T').data_m
#    data.to_csv('./' + category + '/' + tradingm[0:4] + '/' + tradingm + '.csv')


# In[ ]:




