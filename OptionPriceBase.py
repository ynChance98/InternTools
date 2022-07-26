#!/usr/bin/env python
# coding: utf-8

# In[2]:


import datetime
import logging
from collections import namedtuple

Model_param = namedtuple('Model_param', ['forward', 'r', 't', 'strike', 'vol'])
Greeks = namedtuple('Greeks', ['delta', 'vega', 'gamma', 'theta', 'rho'])


class OptionPriceInterface(object):
    """
    期权定价模型接口
    """
    logger = logging.getLogger(__name__)

    def __init__(self):
        self.logger.debug("begin init OptionPriceInterface.")

    def greeks(self, cp):
        """
        返回greeks
        :param cp: None 返回一个list -1:返回put期权的greeks 1：返回call期权的greeks
        :return:
        """
        pass

    def delta(self, cp):
        """
        返回delta
        :param cp: None 返回一个list -1:返回put期权的delta 1：返回call期权的delta
        :return:
        """
        pass

    def vega(self):
        pass

    def gamma(self):
        pass

    def theta(self, cp):
        pass

    def rho(self, cp):
        pass

    def theory_price(self, cp):
        """
        返回理论价格
        :param cp: None 返回一个list -1:返回put期权的价格 1：返回call期权的价格
        :return:
        """
        pass

    def calc_iv(self, option_price, cp):
        pass


class OptionPriceBase(OptionPriceInterface):
    """
    期权定模型基类
    提供了基础时间处理函数
    """
    _day_year = 365
    _hour_year = 24
    _minute_year = 60
    _second_year = 60

    def __init__(self, forward: float, r: float, t: float, strike: float, vol: float):
        self.logger.debug("begin init OptionPriceBase.")
        super().__init__()
        self.const_tradingday_sec_year = (self._day_year * self._hour_year *
                                          self._minute_year * self._second_year)
        assert isinstance(forward, float), "forward need float, get {}".format(type(forward))
        self.forward = forward
        assert isinstance(r, float), "r need float, get {}".format(type(r))
        self.r = r
        assert isinstance(t, float), "t need float, get {}".format(type(t))
        self.t = t
        assert isinstance(strike, float), "strike need float, get {}".format(type(strike))
        self.strike = strike
        assert isinstance(vol, float), "vol need float, get {}".format(type(vol))
        self.vol = vol
        self.logger.debug("end init OptionPriceBase.")

    @property
    def tradingday_year(self):
        return self._day_year

    @tradingday_year.setter
    def tradingday_year(self, day_in_year):
        self.logger.info(f"reset trading day in year to {day_in_year}")
        self._day_year = day_in_year
        self.const_tradingday_sec_year = (self._day_year * self._hour_year *
                                          self._minute_year * self._second_year)

    @property
    def tradinghour_day(self):
        return self._hour_year

    @tradinghour_day.setter
    def tradinghour_day(self, hour_in_day):
        self.logger.info(f"reset trading hour in day to {hour_in_day}")
        self._hour_year = hour_in_day
        self.const_tradingday_sec_year = (self._day_year * self._hour_year *
                                          self._minute_year * self._second_year)

    def measure_vtexp(self, tradingday: str, data_time: str, expiry_date: str, expiry_time='15:00:00') -> float:
        """
        根据当前时间计算 vtexp 精确到s
        :param tradingday: 当前交易日
        :param data_time: 行情数间
        :param expiry_date: 期权到期日
        :param expiry_time: 期权到期日收盘时间
        :return: vtexp 参数
        """
        dt = datetime.datetime.strptime(tradingday + " " + data_time, "%Y%m%d %H:%M:%S")
        ed = datetime.datetime.strptime(expiry_date + " " + expiry_time, "%Y%m%d %H:%M:%S")
        delta_time = (ed - dt).total_seconds()
        if delta_time < 0:
            raise ValueError(f"Get negative time diff. trading day {tradingday} must after expiry day {expiry_date}")
        return delta_time / self.const_tradingday_sec_year


# In[ ]:




