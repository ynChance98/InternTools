#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd

# import empyrical
import shrp_calmar

# 显示所有列
#pd.set_option('display.max_columns', None)
# 显示所有行
#pd.set_option('display.max_rows', None)


# 返回交易信号
# 1 表示long
# -1 表示short
# 0 表示不做操作
def trade_signal(y_pred, long_threshold, short_threshold):
    if y_pred > long_threshold:
        return 1
    if y_pred < short_threshold:
        return -1
    return 0
        
    

# 1 表示胜
# -1 表示输
# 0 表示未操作
def win(y_true, signal):
    if signal == 0:
        return 0
    if (signal == 1 and y_true > 0) or (signal == -1 and y_true < 0):
        return 1
    return -1


def get_threshold(y_train_pred, q):
    df = pd.DataFrame({"y_train_pred": y_train_pred})
    short_threshold = df["y_train_pred"].quantile(q)
    long_threshold = df["y_train_pred"].quantile(1 - q)
    return long_threshold, short_threshold


# y_true 真实值
# y_pred 预测值
# 返回胜率、每单收益、每日报单数
def metric(index, y_true, y_pred, long_threshold, short_threshold):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "long_threshold": long_threshold, "short_threshold": short_threshold}, index=index)
    df["signal"] = df.loc[:,['y_pred','long_threshold','short_threshold']].apply(lambda y: trade_signal(y[0], y[1], y[2]),axis=1)
    df["return"] = df["y_true"] * df["signal"]
    df["win"] = df.apply(lambda r: win(r.y_true, r.signal), axis=1)

    # print(long_threshold, short_threshold)
    # df.to_csv("metric.csv")

    win_rate = len(df.loc[df["win"] == 1]) / len(df.loc[df["win"] != 0])
    long_win_rate = len(df.loc[(df["win"] == 1) & (df["signal"] == 1)]) / len(df.loc[df["signal"] == 1])
    short_win_rate = len(df.loc[(df["win"] == 1) & (df["signal"] == -1)]) / len(df.loc[df["signal"] == -1])

    return_per_order = df["return"].sum() / len(df.loc[df["signal"] != 0])
    long_return_per_order = df.loc[df["signal"] == 1]["return"].sum() / len(df.loc[df["signal"] == 1])
    short_return_per_order = df.loc[df["signal"] == -1]["return"].sum() / len(df.loc[df["signal"] == -1])

    # 计算有多少个交易日
    tradingdays = len(set([x.strftime("%Y%m%d") for x in df.index]))

    order_per_day = len(df.loc[df["signal"] != 0]) / tradingdays
    long_order_per_day = len(df.loc[df["signal"] == 1]) / tradingdays
    short_order_per_day = len(df.loc[df["signal"] == -1]) / tradingdays

    return {"win_rate": win_rate,
            "long_win_rate": long_win_rate, "short_win_rate": short_win_rate,
            "return_per_order": return_per_order,
            "long_return_per_order": long_return_per_order, "short_return_per_order": short_return_per_order,
            "order_per_day": order_per_day,
            "long_order_per_day": long_order_per_day, "short_order_per_day": short_order_per_day}


# 根据信号，计算仓位
def position(row, period):
    for i in range(period):
        if row["signal" + str(i)] != np.NAN and row["signal" + str(i)] != 0:
            return row["signal" + str(i)]
    return 0


# 对预测结果进行回测，得到每个bar的持仓和盈亏
def backtest(index, y1, y_pred, long_threshold, short_threshold, period, fee):
    #df = pd.DataFrame({"y1": y1, "y_pred": y_pred}, index=index)
    #df["signal0"] = df['y_pred'].apply(lambda y: trade_signal(y, long_threshold, short_threshold))
    df = pd.DataFrame({"y1": y1, "y_pred": y_pred, "long_threshold": long_threshold, "short_threshold": short_threshold}, index=index)
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

    return backtest_metric(df),abs(df["signal0"]).sum()


# 根据持仓情况和盈亏计算回测的各项指标
# 输入dataframe，dataframe的index为日期格式
# df中至少包含position、pnl这两列
def backtest_metric(df):
    result = dict()
    df['Tradingday'] = [x.strftime("%Y%m%d") for x in df.index]
    # df['Tradingday'] = df.index
    
    # 计算有多少个交易日
    tradingdays = df['Tradingday'].unique().shape[0]
    # 根据position来算交易次数，一开一平算一次完整的交易
    trade_count = np.abs(df["position"].diff()).sum() / 2
    if trade_count > 0 and tradingdays > 0:
        result["trade_count_per_day"] = trade_count / tradingdays
        result["bps_margin"] = 10000 * df["pnl"].sum() / trade_count
        result["hold_period"] = df.loc[df["position"] != 0].shape[0] / trade_count
    else:
        result["trade_count_per_day"] = 0
        result["bps_margin"] = 0
        result["hold_period"] = 0
    
    # 计算每日收益率
    # s_daily_returns = df["pnl"].resample(rule="1d").sum()
    s_daily_returns = df[['Tradingday','pnl']].groupby('Tradingday').sum()
    # 计算总体的各项指标
    result["sharpe_ratio"], result["calmar_ratio"], result["max_drawdown"], result["annual_return"] = evaluate(s_daily_returns)

    # 计算多头每日收益率
    df_long = df.copy()
    df_long.loc[df_long["position"] < 0, "pnl"] = 0
    df_long.loc[df_long["position"] < 0, "position"] = 0
    long_trade_count = df_long["position"].diff().abs().sum() / 2
    if long_trade_count > 0 and tradingdays > 0:
        result["long_trade_count_per_day"] = long_trade_count / tradingdays
        result["long_bps_margin"] = 10000 * df_long["pnl"].sum() / long_trade_count
        result["long_hold_period"] = df_long.loc[df_long["position"] != 0].shape[0] / long_trade_count
    else:
        result["long_trade_count_per_day"] = 0
        result["long_bps_margin"] = 0
        result["long_hold_period"] = 0

    # s_long_daily_returns = df_long["pnl"].resample(rule="1d").sum()
    s_long_daily_returns = df_long[['Tradingday','pnl']].groupby('Tradingday').sum()
    
    # 计算多头的各项指标
    result["long_sharpe_ratio"], result["long_calmar_ratio"], result["long_max_drawdown"], result["long_annual_return"] = evaluate(s_long_daily_returns)

    # 计算空头每日收益率
    df_short = df.copy()
    df_short.loc[df_short["position"] > 0, "pnl"] = 0
    df_short.loc[df_short["position"] > 0, "position"] = 0
    short_trade_count = df_short["position"].diff().abs().sum() / 2
    if short_trade_count > 0 and tradingdays > 0:
        result["short_trade_count_per_day"] = short_trade_count / tradingdays
        result["short_bps_margin"] = 10000 * df_short["pnl"].sum() / short_trade_count
        result["short_hold_period"] = df_short.loc[df_short["position"] != 0].shape[0] / short_trade_count
    else:
        result["short_trade_count_per_day"] = 0
        result["short_bps_margin"] = 0
        result["short_hold_period"] = 0

    # s_short_daily_returns = df_short["pnl"].resample(rule="1d").sum()
    s_short_daily_returns = df_short[['Tradingday','pnl']].groupby('Tradingday').sum()
    
    # 计算空头的各项指标
    result["short_sharpe_ratio"], result["short_calmar_ratio"], result["short_max_drawdown"], result["short_annual_return"] = evaluate(s_short_daily_returns)

    # 单独计算每一年的各种指标
    begin_year = df.index[0].strftime("%Y")
    end_year = df.index[-1].strftime("%Y")
    while begin_year <= end_year:
        begin_tradingday = begin_year + "0101"
        end_tradingday = begin_year + "1231"
        df_year = df.loc[begin_tradingday:end_tradingday]
        tradingdays = len(set([x.strftime("%Y%m%d") for x in df_year.index]))
        trade_count = df_year["position"].diff().abs().sum() / 2
        if trade_count > 0 and tradingdays > 0:
            result["trade_count_per_day_" + begin_year] = trade_count / tradingdays
            result["bps_margin_" + begin_year] = 10000 * df_year["pnl"].sum() / trade_count
            result["hold_period_" + begin_year] = df_year.loc[df_year["position"] != 0].shape[0] / trade_count

        else:
            result["trade_count_per_day_" + begin_year] = 0
            result["bps_margin_" + begin_year] = 0
            result["hold_period_" + begin_year] = 0

        # s_returns = df_year["pnl"].resample(rule="1d").sum()
        s_returns = df_year[['Tradingday','pnl']].groupby('Tradingday').sum()
        result["sharpe_ratio_" + begin_year], result["calmar_ratio_" + begin_year], result[
            "max_drawdown_" + begin_year], result["annual_return_" + begin_year] = evaluate(s_returns)

        begin_year = str(int(begin_year) + 1)
    return result,s_returns


# 根据每日盈亏，计算sharpe_ratio、calmar_ratio、max_drawdown、annual_return等指标
# 输入s_daily_returns为每日simple return，非累计
def evaluate(s_daily_returns):
    s_daily_returns_cum = s_daily_returns.cumsum()
    # sharpe_ratio = empyrical.sharpe_ratio(s_daily_returns)
    # calmar_ratio = empyrical.calmar_ratio(s_daily_returns)
    # max_drawdown = empyrical.max_drawdown(s_daily_returns)
    # annual_return = empyrical.annual_return(s_daily_returns)
    sharpe_ratio = shrp_calmar.calc_Sharpe(s_daily_returns.index, s_daily_returns.values)
    calmar_ratio, max_drawdown, _, _, _ = shrp_calmar.calc_Calmar(s_daily_returns_cum.index, s_daily_returns_cum.values.squeeze())
    annual_return = shrp_calmar.calc_AnnRtn(s_daily_returns_cum.index, s_daily_returns_cum.values.squeeze())
    return sharpe_ratio, calmar_ratio, max_drawdown, annual_return


if __name__ == '__main__':
    df = pd.DataFrame({"y_true": np.random.rand(10000),
                       "y_pred": np.random.rand(10000)},
                      index=pd.date_range("20111231", periods=10000, freq="60s"))
    # print(df.index[0])
    # print(df.index[-1])
    # print(metric(df.index, df["y_true"], df["y_pred"], 0.7, 0.3))
    print(backtest(df.index, df["y_true"], df["y_pred"], 0.7, 0.3, 3, 0.1))


# In[ ]:
def backtest_min(index, y1, y_pred, long_threshold, short_threshold, period, fee):
    #df = pd.DataFrame({"y1": y1, "y_pred": y_pred}, index=index)
    #df["signal0"] = df['y_pred'].apply(lambda y: trade_signal(y, long_threshold, short_threshold))
    df = pd.DataFrame({"y1": y1, "y_pred": y_pred, "long_threshold": long_threshold, "short_threshold": short_threshold}, index=index)
    df["signal0"] = df.loc[:,['y_pred','long_threshold','short_threshold']].apply(lambda y: trade_signal(y[0], y[1], y[2]),axis=1)
    for i in range(period - 1):
        df["signal" + str(i + 1)] = df["signal0"].shift(i + 1)
    df["position"] = df.apply(lambda row: position(row, period), axis=1)

    # 收益
    df["return"] = df["position"] * df["y1"]
    # 手续费
    df["fee"] = abs(df["position"].diff()) * fee
    df["fee"] = df["fee"].fillna(0)
    # 盈亏 = 收益 - 手续费
    df["pnl"] = (df["return"] - df["fee"]) #/ 10000

    return df




