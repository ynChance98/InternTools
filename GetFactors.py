from metrics import * 
import pandas as pd
from shrp_calmar import * 
import numpy as np
from scipy import stats
import warnings 
from BTtools import *


class ready_GetF(object):
    def __init__(self, data_y, t_Colname):
        self.t_Colname = t_Colname
        self.data_y = data_y.set_index(self.t_Colname)
        self.time_list = data_y.index.unique().tolist()
        self.date_list = data_y.date.unique().tolist()
        
    def Basic_tick_f(self):  # 生成tick级别的基础数据和基础factor
        total = []
        for date in self.time_list:
            data = self.data_y.loc[date]
            tlist = np.unique(data.t)
            t_list=list(tlist)
            strikelist = np.unique(data.strike)
            strike_list= list(strikelist)
            x = data.groupby(['t', 'strike', 'cp'])
            result = []
            for t in t_list:
                for strike in strike_list:
                    try:
                        call = x.get_group((t, strike, 1))
                        put = x.get_group((t, strike, -1))
                        call['volume_p'] = put.volume.to_list()
                        call['openinterest_p'] = put.openinterest.to_list()
                        call['opt_price_p'] = put.opt_price.to_list()
                        call['iv_p'] = put.iv.to_list()
                        call['volume_pcr'] = call.volume_p / call.volume
                        call['openinterest_pcr'] = call.openinterest_p / call.openinterest
                        call['skew'] = call.iv_p / call.iv
                        #call['rv'] = ((np.log(put.forward)-np.log(put.forward.shift())).dropna()**2).sum() ### ？
                        call = call.rename(columns={'iv': 'iv_c', 'volume': 'volume_c', 'openinterest': 'openinterest_c', 'opt_price': 'opt_price_c'})
                        result.append(call)
                    except:
                        continue
            try:
                result = pd.concat(result)
            except:
                continue
            total.append(result)
        total = pd.concat(total)
        total['ctp_c'] = total.volume_c * total.strike / total.groupby([total.index]).volume_c.transform('sum')
        total['ctp_p'] = total.volume_p * total.strike / total.groupby([total.index]).volume_p.transform('sum')
        total['td'] = total.ctp_c - total.ctp_p
        total['td_pcr']=(total.volume_c*total.opt_price_c)/(total.volume_p*total.opt_price_p)
        total['proportion'] = total.strike / total.forward
        total['gap'] = abs(total.strike - total.forward)
        return total
    
    def Basic_Mins_f(self):#从上面tick级别的数据再生成分钟级别
        total = self.Basic_tick_f().dropna(subset=['forward'])
        total['volume_day'] = total.groupby(['date', 'expirydate']).volume_c.transform('sum') + total.groupby(['date', 'expirydate']).volume_p.transform('sum')
        total['volume_day_max'] = total.groupby('date').volume_day.transform('max')
        main = total[total.volume_day == total.volume_day_max]
        main['rank'] = main.groupby(['updatetime', 'date'])['gap'].rank(method='first').astype(int)
        atm = main[main['rank'] <= 1].sort_index()
        date_list = atm.date.unique().tolist()
        atm['rv'] = ((np.log(atm.forward)-np.log(atm.forward.shift())).dropna()**2)#.sum()
        return atm
    
 # 以下为生成各种factor的函数
    
    def Surplus(self, time_colums ):
        rv = []                     
        iv_c_d = []              
        iv_p_d = []
        skew_day = []
        atm = self.Basic_Mins_f()
        z = atm.groupby(time_colums)#分钟级别time的列名
        time_list = atm.index.unique().tolist()
        for date in time_list:  
            cut = z.get_group(date)
            rv.append(np.mean(cut['rv']))
            iv_c_d.append(np.mean(cut['iv_c']))
            iv_p_d.append(np.mean(cut['iv_p']))
            cut = cut[-np.isinf(cut['skew'])].dropna()
            skew_day.append(np.mean(cut['skew']))
        rv = Standardscaler(rv)
        iv_c_d = Standardscaler(iv_c_d)
        iv_p_d = Standardscaler(iv_p_d)
        surplus_c = iv_c_d - rv
        surplus_p = iv_p_d - rv
        z = atm.groupby('date')
        return_day = []
        for date in date_list:
            day = z.get_group((date))
            return_ = (day.iloc[-1, 8] - day.iloc[0, 8]) / day.iloc[0, 8]
            return_day.append(return_)
        return_day = pd.DataFrame(return_day)
        z = atm.groupby('time')
        return_day_list = []
        day_list = []
        for date in time_list:
            day = z.get_group((date))
            return_ = (day.iloc[-1, 8] - day.iloc[0, 8]) / day.iloc[0, 8]
            return_day_list.append(return_)
            day_list.append(day.iat[0, 1])
        return pd.DataFrame({'date': time_list, 'y_rtn': return_day_list, 'rv': rv, 'iv_c': iv_c_d, 'iv_p': iv_p_d, 'dirvc': surplus_c, 'dirvp': surplus_p}).set_index('date')

    def TDivrgence(self, time_colums ):
        td = []
        atm = self.Basic_Mins_f()
        z = atm.groupby(time_colums)
        time_list = atm.index.unique().tolist()
        for date in time_list:
            cut = z.get_group(date)
            td.append(np.mean(cut['td']))
        td = Standardscaler(td)
        return_day_list = []
        day_list = []
        for date in time_list:
            day = z.get_group((date))
            return_ = (day.iloc[-1, 8] - day.iloc[0, 8]) / day.iloc[0, 8]
            return_day_list.append(return_)
            day_list.append(day.iat[0, 1])
        return pd.DataFrame({'date': time_list, 'y_rtn': return_day_list,'td': td}).set_index('date')
    
    def TD_PCR(self, time_colums ):
        atm = self.Basic_Mins_f()
        time_list = atm.index.unique().tolist()
        td_pcr = []
        z = atm.groupby(time_colums)
        for date in time_list:
            cut = z.get_group(date)
            td_pcr.append((cut.volume_c*cut.opt_price_c).sum()/(cut.volume_p*cut.opt_price_p).sum())
        td_pcr = Standardscaler(td_pcr)
        return_day_list = []
        day_list = []
        for date in time_list:
            day = z.get_group((date))
            return_ = (day.iloc[-1, 8] - day.iloc[0, 8]) / day.iloc[0, 8]
            return_day_list.append(return_)
            day_list.append(day.iat[0, 1])
        return pd.DataFrame({'date': time_list, 'y_rtn': return_day_list,'td_pcr': td_pcr}).set_index('date')

    def CP_skew(self, time_colums ):
        atm = self.Basic_Mins_f()
        time_list = atm.index.unique().tolist()
        # 导出标的收益率 出因子值 cpskew
        return_day_list = []
        day_list = []
        skew_day = []
        z = atm.groupby(time_colums)

        for date in time_list:
            cut = z.get_group(date)
            cut = cut[-np.isinf(cut['skew'])].dropna()
            skew_day.append(np.mean(cut['skew']))
            day = z.get_group((date))
            return_ = (day.iloc[-1, 8] - day.iloc[0, 8]) / day.iloc[0, 8]
            return_day_list.append(return_)
            day_list.append(day.iat[0, 1])
        return pd.DataFrame({'date': time_list, 'y_rtn': return_day_list, 'skew': skew_day}).set_index('date')
    
    def OI_PCR(self, time_colums ):
        atm = self.Basic_Mins_f()
        time_list = atm.index.unique().tolist()
        oi_pcr = []
        z = atm.groupby(time_colums)
        for date in time_list:
            cut = z.get_group(date)
            oi_pcr.append(np.mean(cut['openinterest_pcr']))
        oi_pcr = Standardscaler(oi_pcr)
        return_day_list = []
        day_list = []
        for date in time_list:
            day = z.get_group((date))
            return_ = (day.iloc[-1, 8] - day.iloc[0, 8]) / day.iloc[0, 8]
            return_day_list.append(return_)
            day_list.append(day.iat[0, 1])
        return pd.DataFrame({'date': time_list, 'y_rtn': return_day_list, 'oi_pcr': oi_pcr}).set_index('date')
    
    
    def allkindsPCR(self):
        total = self.Basic_tick_f()
        total['timee']=total.index
        total['timee']=total['timee'].apply(lambda x : x[:16])
        total['c_amount']=total.volume_c*total.opt_price_c
        total['p_amount']=total.volume_p*total.opt_price_p
        pcr_all = total[['volume_c','openinterest_c', 'volume_p', 'openinterest_p',  'timee',
       'c_amount', 'p_amount']].groupby('timee').sum()
        pcr_all['APCR'] = pcr_all.c_amount/pcr_all.p_amount
        pcr_all['OPCR'] = pcr_all.openinterest_c/pcr_all.openinterest_p
        pcr_all['VPCR'] = pcr_all.volume_c/pcr_all.volume_p

        pcr_all['volume_c_20min'] = pcr_all.volume_c.rolling(20).sum()
        pcr_all['volume_p_20min'] = pcr_all.volume_p.rolling(20).sum()
        pcr_all['R20VPCR']=pcr_all['volume_c_20min']/pcr_all['volume_p_20min']

        pcr_all['openinterest_c_20min'] = pcr_all.openinterest_c.rolling(20).sum()
        pcr_all['openinterest_p_20min'] = pcr_all.openinterest_p.rolling(20).sum()
        pcr_all['R20OPCR']=pcr_all['openinterest_c_20min']/pcr_all['openinterest_p_20min']

        pcr_all['c_amount_20min'] = pcr_all.c_amount.rolling(20).sum()
        pcr_all['p_amount_20min'] = pcr_all.p_amount.rolling(20).sum()
        pcr_all['R20APCR']=pcr_all['c_amount_20min']/pcr_all['p_amount_20min']

        pcr_all['volume_c_5min'] = pcr_all.volume_c.rolling(5).sum()
        pcr_all['volume_p_5min'] = pcr_all.volume_p.rolling(5).sum()
        pcr_all['R5VPCR']=pcr_all['volume_c_5min']/pcr_all['volume_p_5min']

        pcr_all['openinterest_c_5min'] = pcr_all.openinterest_c.rolling(5).sum()
        pcr_all['openinterest_p_5min'] = pcr_all.openinterest_p.rolling(5).sum()
        pcr_all['R5OPCR']=pcr_all['openinterest_c_5min']/pcr_all['openinterest_p_5min']

        pcr_all['c_amount_5min'] = pcr_all.c_amount.rolling(5).sum()
        pcr_all['p_amount_5min'] = pcr_all.p_amount.rolling(5).sum()
        pcr_all['R5APCR']=pcr_all['c_amount_5min']/pcr_all['p_amount_5min']
        return pcr_all
    
    def ATMAPCR(self, n): #n是atm附近n个合约
        total = self.Basic_tick_f()
        total = total.dropna(subset=['forward'])
        total['volume_day'] = total.groupby(['date', 'expirydate']).volume_c.transform('sum') + total.groupby(['date', 'expirydate']).volume_p.transform('sum')
        total['volume_day_max'] = total.groupby('date').volume_day.transform('max')
        main = total[total.volume_day == total.volume_day_max]
        main['rank'] = main.groupby(['updatetime', 'date'])['gap'].rank(method='first').astype(int)
        atm_apcr = main[main['rank'] <= n].sort_index()
        atm_apcr['timee'] = atm_apcr.index
        atm_apcr['timee']= atm_apcr['timee'].apply( lambda x : x[:16])
        atm_apcr['amount_c']= atm_apcr.volume_c * atm_apcr.opt_price_c
        atm_apcr['amount_p']=atm_apcr.volume_p * atm_apcr.opt_price_p
        ATMAPCR = atm_apcr.groupby('timee')[['amount_c','amount_p']].sum()
        ATMAPCR['ATMAPCR']=ATMAPCR.amount_c/ATMAPCR.amount_p
        ATMAPCR['c_amount_5min'] = ATMAPCR.amount_c.rolling(5).sum()#滚动5 分钟的ATMAPCR
        ATMAPCR['p_amount_5min'] = ATMAPCR.amount_p.rolling(5).sum()
        ATMAPCR['ATMR5APCR']=ATMAPCR['c_amount_5min']/ATMAPCR['p_amount_5min']
        ATMAPCR['c_amount_20min'] = ATMAPCR.amount_c.rolling(20).sum()
        ATMAPCR['p_amount_20min'] = ATMAPCR.amount_p.rolling(20).sum()
        ATMAPCR['ATMR20APCR']=ATMAPCR['c_amount_20min']/ATMAPCR['p_amount_20min']
        ATMAPCR = ATMAPCR[['ATMAPCR','ATMR5APCR','ATMR20APCR']]
        return ATMAPCR



    
    
    
    
    
    
    
    
    
    
    