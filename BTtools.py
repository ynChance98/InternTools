from metrics import * 
import pandas as pd
from shrp_calmar import * 
import numpy as np
from scipy import stats


def get_bolling(data, mul, length):  # 生成布林带
    _std = data.rolling(length,min_periods=1).std()
    _mean = data.rolling(length,min_periods=1).mean()
    return (_mean + mul * _std).fillna(0).tolist(), (_mean - mul * _std).fillna(0).tolist()

simple_key_list = ['annual_return', 'sharpe_ratio', 'calmar_ratio', 'max_drawdown', 'bps_margin', 'hold_period'] 
# print('|'.join(simple_key_list))

def result_to_str(rst: dict, kl: list, sep:str='|') -> str:
    assert set(kl) <= set(rst.keys())
    ss = ''
    for k in kl:
        ss += str(rst[k]) + sep
    return ss[0:-1]


def Standardscaler(data,period=50):  #带有滚动窗口的数据标准化函数
    data = pd.DataFrame(data)
    #data[np.isinf(data[0])] = 0
    data=data.fillna(method='ffill')
    data = (data.iloc[:,-1] - data.iloc[:,-1].rolling(period,min_periods=1).mean())/data.iloc[:,-1].rolling(period,min_periods=1).std().fillna(method='bfill')
    return data

def sigmoid(x):
    return 1/(np.exp(-x)+1)

def get_IC2(yreturn,factorvalues):
    return yreturn.corrwith(factorvalues,method='pearson')

def BTGA(index, y1, y_pred, long_threshold, short_threshold, period, fee): #用于遗传算法的回测工具
    const_dd = len(y1)/255  
    df = pd.DataFrame({"y1": y1, "y_pred": y_pred.values, "long_threshold": long_threshold, "short_threshold": short_threshold}, index=index)
    df["signal0"] = df.loc[:,['y_pred','long_threshold','short_threshold']].apply(lambda y: trade_signal(y[0], y[1], y[2]),axis=1)
    for i in range(period - 1):
        df["signal" + str(i + 1)] = df["signal0"].shift(i + 1)
    df["position"] = df.apply(lambda row: position(row, period), axis=1)

    # 收益
    df["return"] = df["position"] * df["y1"]
    #冲击成本
    # 手续费
    df["fee"] = abs(df["position"].diff()) * fee
    df["fee"] = df["fee"].fillna(0)
    # 盈亏 = 收益 - 手续费
    df["pnl"] = (df["return"] - df["fee"]) #/ 10000
    pnlcum = df.pnl.cumsum()
    Annual =  pnlcum[-1] / const_dd  
    sharp = math.sqrt(163) * df.pnl.mean() / df.pnl.std()
    BPS = pnlcum[-1]/(abs(df.signal0).sum())
    
    return Annual,sharp,BPS