#!/usr/bin/env python
# coding: utf-8

# In[3]:


import math
import numpy as np


def calc_AnnRtn(TradingDay, pnl_cum):
    if len(pnl_cum) == 0:
        return np.nan
    n_day = len(set(TradingDay))
    return pnl_cum[-1] / (n_day / 244)


def calc_Sharpe(TradingDay, pnl):
    if pnl.std() == 0 or np.isnan(pnl.std()):
        return 0
    else:
        return math.sqrt(244) * pnl.mean() / pnl.std()


def calc_Calmar(TradingDay, pnl_cum):
    top_pnl = 0
    valley_pnl = 0
    mdd = 0
    mddtime = 0
    mddtime_tmp = 0
    i_mdd_start = 0
    mddbegin = 0
    mddend = 0
    n_day = len(set(TradingDay))
    for i in range(pnl_cum.shape[0]):
        if pnl_cum[i] >= top_pnl:
            top_pnl = pnl_cum[i]
            valley_pnl = pnl_cum[i]
            mddtime_tmp = 0
            i_mdd_start = i
        else:
            mddtime_tmp += 1
            valley_pnl = min(valley_pnl, pnl_cum[i])
            if top_pnl - valley_pnl > mdd:
                mdd = top_pnl - valley_pnl
                mddtime = mddtime_tmp
                mddbegin = TradingDay[i_mdd_start]
                mddend = TradingDay[i]
    if mdd:
        Calmar = pnl_cum[-1] / mdd / (n_day / 244)
    else:
        Calmar = 0
    return Calmar, mdd, mddtime, mddbegin, mddend


# In[ ]:




