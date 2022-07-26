#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import math
from scipy.stats import norm
from DataFrameBase import DataFrameBase
from OptionPriceBase import OptionPriceBase, Greeks


class BsmPy(OptionPriceBase):
    """
    应用Black-Scholes Model 实现 OptionPriceBase
    """

    def __init__(self, forward: float, r: float, t: float, strike: float, vol: float):
        super().__init__(forward, r, t, strike, vol)
        self.sqt = math.sqrt(self.t)
        self.norm_pdf_d1, self.norm_cdf_d1, self.norm_cdf_d2 = self.__get_d(self.forward, self.vol)

    def __get_d(self, forward: float, vol: float) -> tuple:
        d1 = (math.log(forward / self.strike) + (self.r + pow(vol, 2) / 2) * self.t) / (vol * self.sqt)
        d2 = d1 - vol * self.sqt
        return norm.pdf(d1, 0, 1), norm.cdf(d1, 0, 1), norm.cdf(d2, 0, 1)

    def delta(self, cp):
        if cp == 1:
            return self.norm_cdf_d1
        elif cp == -1:
            return self.norm_cdf_d1 - 1
        else:
            raise ValueError(f"get illegal parameter {cp}")

    def vega(self):
        return self.forward * self.norm_pdf_d1 * self.sqt

    def gamma(self):
        return self.norm_pdf_d1 / (self.forward * self.vol * self.sqt)

    def theta(self, cp):
        if abs(cp) == 1:
            return -1 * (self.forward * self.vol * self.norm_pdf_d1) / (2 * self.sqt) -                    cp * self.r * self.strike * math.exp(-1 * self.r * self.t) * (
                           (self.norm_cdf_d2 - 0.5) * cp + 0.5)
        else:
            raise ValueError(f"Get illegal parameter {cp}")

    def rho(self, cp):
        if abs(cp) == 1:
            return cp * self.strike * self.t * ((self.norm_cdf_d2 - 0.5) * cp + 0.5) *                    math.exp(-1 * self.r * self.t)
        else:
            raise ValueError(f"Get illegal parameter {cp}")

    def greeks(self, cp: int) -> Greeks:
        if abs(cp) == 1:
            return Greeks(self.delta(cp), self.vega(), self.gamma(), self.theta(cp), self.rho(cp))
        else:
            raise ValueError(f"Get illegal parameter {cp}")

    def theory_price(self, cp: int) -> float:
        if abs(cp) == 1:
            return self.__theory_price(cp, self.norm_cdf_d1, self.norm_cdf_d2)
        else:
            raise ValueError(f"Get illegal parameter {cp}")

    def __theory_price(self, cp: int, cdf1, cdf2) -> float:
        x = self.forward * ((cdf1 - 0.5) * cp + 0.5)
        y = self.strike * math.exp(-1 * self.r * self.t) * ((cdf2 - 0.5) * cp + 0.5)
        return cp * (x - y)

    def calc_iv(self, option_price: float, cp: int) -> float:
        check = self.__check_newton(cp, option_price, 2)
        if check is None:
            vol_temp = self.__iv_li(option_price)
            if vol_temp is None:
                temp = self.__iv_newton(cp, option_price, 0.5)
                if temp is None:
                    temp = self.__iv_newton(cp, option_price, 1)
                    if temp is None:
                        return self.__iv_newton(cp, option_price, 1.5)
                    else:
                        return temp
                else:
                    return temp
            else:
                return self.__iv_newton(cp, option_price, vol_temp)
        else:
            return check

    def __iv_li(self, option_price):
        sqrt2pi = 2.506628274
        sqrt32 = 5.656854249
        sqrt2_2 = 2.828427124
        sqrt2_6 = 4.242640687
        enrt = math.exp(-1 * self.r * self.t)
        eta = self.strike * enrt / self.forward  # 有问题
        rho = abs(self.strike * enrt - self.forward) * self.forward / (option_price * option_price)  # ???
        alpha = (2 * option_price / self.forward + eta - 1) * sqrt2pi / (1 + eta)
        beta = math.cos(math.acos(3 * alpha / sqrt32) / 3)

        try:
            if rho <= 1.4:
                return sqrt2_2 * beta / self.sqt -                        math.sqrt((8 * beta * beta - sqrt2_6 * alpha / beta) / self.t)
            else:
                return (alpha + math.sqrt(alpha * alpha - (4 * (eta - 1) * (eta - 1) / (eta + 1)))) / (2 * self.sqt)
        except ValueError:
            return None

    def __check_newton(self, cp, option_price, max_vol):
        pdf, cdf1, cdf2 = self.__get_d(self.forward, 0.00001)
        f1 = self.__theory_price(cp, cdf1, cdf2) - option_price
        if f1 > 0:
            return 0
        pdf, cdf1, cdf2 = self.__get_d(self.forward, max_vol)
        f2 = self.__theory_price(cp, cdf1, cdf2) - option_price
        if f2 < 0:
            return max_vol
        return None

    def __iv_newton(self, cp: int, option_price: float, vol_init: float):
        pdf, cdf1, cdf2 = self.__get_d(self.forward, vol_init)
        f1 = self.__theory_price(cp, cdf1, cdf2) - option_price
        if 0.00001 > f1 > -0.00001:
            return vol_init
        else:
            f2 = self.forward * self.sqt * pdf
            if 0.00001 > f2 > -0.00001:
                return None  # change the init of  iv to 1 or 1.5
            else:
                vol_new = vol_init - (f1 / f2)
                return self.__iv_newton(cp, option_price, vol_new)


class BsmMatrix(DataFrameBase, OptionPriceBase):
    """
    这个类应该被继承，指定_rename_dict 用来转换列名
    建议使用instrumentID作为索引
    """
    _head = ('opt_price', 'cp', 'forward', 'r', 't', 'strike', 'vol')
    _rename_dict = {}
    # _rename_dict = {'opt_price': 'opt_price', 'cp': 'cp', 'forward': 'forward', 'r': 'r', 't': 't', 'strike': 'strike',
    #                 'vol': 'vol'}
    _var_type_dict = {int: ['cp'], float: ['opt_price', 'forward', 'r', 't', 'strike', 'vol'], str: []}

    def __init__(self, df):
        super(BsmMatrix, self).__init__(df=df)
        self._df['sqt'] = np.sqrt(self._df.t)
        self._df['norm_pdf_d1'], self._df['norm_cdf_d1'], self._df['norm_cdf_d2'] =             self.__get_d(self._df.forward, self._df.vol)
        self._df['m'] = 1
        self._df['init'] = 0
        self._df['iv'] = 0
        self._df['f1'] = 0
        self._df['f2'] = 0

    def __get_d(self, forward: pd.Series, vol):
        d1 = (np.log(forward / self._df.strike) + (self._df.r + vol * vol / 2) * self._df.t) /              (vol * self._df.sqt)
        d2 = d1 - vol * self._df.sqt
        return norm.pdf(d1), norm.cdf(d1), norm.cdf(d2)

    @staticmethod
    def get_d1(forward, strike, r, t, sqt, vol):
        d1 = (np.log(forward / strike) + (r + vol * vol / 2) * t) /              (vol * sqt)
        d2 = d1 - vol * sqt
        return norm.pdf(d1), norm.cdf(d1), norm.cdf(d2)

    def delta(self, cp=None):
        """

        :param cp: 使用内部自带属性，因此该参数无效
        :return:
        """
        return self._df.norm_cdf_d1 + (self._df.cp * 0.5 - 0.5)

    def vega(self):
        return self._df.forward * self._df.norm_pdf_d1 * self._df.sqt

    def gamma(self):
        return self._df.norm_pdf_d1 / (self._df.forward * self._df.vol * self._df.sqt)

    def theta(self, cp=None):
        return (-1 * (self._df.forward * self._df.vol * self._df.norm_pdf_d1) / (2 * self._df.sqt) -
                self._df.cp * self._df.r * self._df.strike * np.exp(-1 * self._df.r * self._df.t) *
                ((self._df.norm_cdf_d2 - 0.5) * self._df.cp + 0.5))

    def rho(self, cp=None):
        return (self._df.cp * self._df.strike * self._df.t * ((self._df.norm_cdf_d2 - 0.5) * self._df.cp + 0.5) *
                np.exp(-1 * self._df.r * self._df.t))

    def theory_price(self, cp=None):
        return self.__theory_price(self._df.norm_cdf_d1, self._df.norm_cdf_d2)

    def __theory_price(self, cdf1: pd.Series, cdf2: pd.Series) -> pd.Series:
        x = self._df.forward * ((cdf1 - 0.5) * self._df.cp + 0.5)
        y = (self._df.strike * np.exp(-1 * self._df.r * self._df.t) *
             ((cdf2 - 0.5) * self._df.cp + 0.5))
        return self._df.cp * (x - y)

    @staticmethod
    def theory_price1(cdf1, cdf2, forward, cp, strike, r, t):
        x = forward * ((cdf1 - 0.5) * cp + 0.5)
        y = strike * np.exp(-1 * r * t) * ((cdf2 - 0.5) * cp + 0.5)
        return cp * (x - y)

    def greeks(self, cp=None):
        temp_df = pd.DataFrame()
        temp_df['delta'] = self.delta()
        temp_df['vega'] = self.vega()
        temp_df['gamma'] = self.gamma()
        temp_df['theta'] = self.theta()
        temp_df['rho'] = self.rho()
        temp_df['theory_price'] = self.theory_price()
        return temp_df

    def calc_iv(self, option_price=None, cp=None) -> pd.Series:
        self.__check_newton(2)
        self._df = self.__iv_newton(0.5)
        self._df.m = self._df.m.replace(2, 1)
        self._df = self.__iv_newton(1)
        self._df.m = self._df.m.replace(2, 1)
        self._df = self.__iv_newton(1.5)
        self._df = self._df.drop(['vol', 'sqt', 'norm_pdf_d1', 'norm_cdf_d1', 'norm_cdf_d2', 'm', 'init', 'f1', 'f2'],
                                 axis=1)
        return self._df

    def __check_newton(self, max_vol):
        pdf, cdf1, cdf2 = self.__get_d(self._df.forward, 0.00001)
        self._df.f1 = self.__theory_price(cdf1, cdf2) - self._df.opt_price
        self._df.loc[self._df.f1 > 0, 'm'] = 0
        self._df.loc[self._df.f1 > 0, 'iv'] = 0
        pdf, cdf1, cdf2 = self.__get_d(self._df.forward, max_vol)
        self._df.f2 = self.__theory_price(cdf1, cdf2) - self._df.opt_price
        self._df.loc[self._df.f2 < 0, 'm'] = 0
        self._df.loc[self._df.f2 < 0, 'iv'] = max_vol

    def __iv_newton(self, init):
        if init == 0.5:
            self._df.loc[(self._df.m == 1) & (self._df.init == 0), 'init'] = 0.5
        else:
            self._df.loc[(self._df.m == 1), 'init'] = init
        result = self._df[self._df.m == 0]
        cal = self._df[self._df.m == 1]
        done = pd.DataFrame()
        while not cal.empty:
            cal['pdf'], cal['cdf1'], cal['cdf2'] = self.get_d1(cal.forward, cal.strike, cal.r, cal.t, cal.sqt, cal.init)
            cal.f1 = self.theory_price1(cal.cdf1, cal.cdf2, cal.forward, cal.cp, cal.strike, cal.r, cal.t) - cal.opt_price
            cal.f2 = cal.forward * cal.sqt * cal.pdf
            cal = cal.fillna(0)
            cal.loc[(cal.f1 < 0.00001) & (cal.f1 > -0.00001), 'iv'] = cal.loc[(cal.f1 < 0.00001) & (cal.f1 > -0.00001), 'init']
            cal.loc[(cal.f2 < 0.00001) & (cal.f2 > -0.00001), 'm'] = 2  # need to change init
            cal.loc[(cal.f1 < 0.00001) & (cal.f1 > -0.00001), 'm'] = 0
            done = done.append(cal[cal.m != 1])
            cal = cal[cal.m == 1]
            cal.init = cal.init - cal.f1 / cal.f2
        result = result.append(done)
        result = result.sort_index()
        return result


"""
test
"""

# import time
# data = pd.read_csv('./iv.csv')
# data = data.rename(columns={'c': 'opt_price'})
# data['vol'] = 0.5
# time_start = time.time()
# data['iv_m'] = BsmMatrix(data[['vol', 'cp', 'r', 't', 'opt_price', 'strike', 'forward']]).calc_iv().iv
# time_end = time.time()
# print('time cost', time_end - time_start, 's')
# data['diff'] = data.iv - data.iv_m
# greek = BsmMatrix(data[['vol', 'cp', 'r', 't', 'opt_price', 'strike', 'forward']]).greeks()
# data = pd.concat([data, greek], axis=1)
# data = data.sample(n=10000)
# data['delta1'] = data.apply(lambda row: BsmPy(row.forward, row.r, row.t, row.strike, row.vol).delta(row.cp), axis=1)
# data['vega1'] = data.apply(lambda row: BsmPy(row.forward, row.r, row.t, row.strike, row.vol).vega(), axis=1)
# data['gamma1'] = data.apply(lambda row: BsmPy(row.forward, row.r, row.t, row.strike, row.vol).gamma(), axis=1)
# data['theta1'] = data.apply(lambda row: BsmPy(row.forward, row.r, row.t, row.strike, row.vol).theta(row.cp), axis=1)
# data['rho1'] = data.apply(lambda row: BsmPy(row.forward, row.r, row.t, row.strike, row.vol).rho(row.cp), axis=1)
#
# data = pd.read_csv('./data.csv')
# data['iv_diff'] = data.iv - data.iv_m
# data['delta_diff'] = data.delta - data.delta1
# data['vega_diff'] = data.vega - data.vega1
# data['gamma_diff'] = data.gamma - data.gamma1
# data['theta_diff'] = data.theta - data.theta1
# data['rho_diff'] = data.rho - data.rho1


# In[ ]:




