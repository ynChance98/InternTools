#Lee Ready algorithm 生成带有交易方向的volumn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import Option
import os
import bsm
import calendar
import datetime as dt

''' 
LeeReadyGen在GetmidData这一步中使用，可以得到每日每分钟每个合约的买卖成交量
:param datapath: 数据目录
:param outpathLR：生成数据目录
:param newfilelist: 每日的data list
'''
class LeeReadyGen(object):
    self.datapath = datapath
    self.outpathLR = outpathLR
    self.newfilelist = newfilelist =[x.name for x in Path(datapath).iterdir() if x.is_dir()] #原datapath中还有别的文件，这一步是为了筛掉别的文件，之后根据情况不同这一步再做更改
    def iolist(self,subfilelist):
        iolist = list()
        for a in subfilelist:
            if a[:2]=='IO':
                iolist.append(a)
        return iolist
    def LR_detail(self,i,ii):
        print(ii)
        iodata=pd.read_csv(self.datapath+i+'/'+ii)
        iodata['minute']=iodata.update_time.apply(lambda x:x[:5])
        iodata.last_price=iodata.last_price.apply(lambda x : float(x))

        iodata['volumm']=iodata.volume.diff()
        iodata['mid_price']=(iodata.ask_price1+iodata.bid_price1)/2
        iodata['buy_sell']=''
        iodata['buy_vol']=0
        iodata['sell_vol']=0
        iodata['transaction']= False

        null_index = iodata.volumm < 1
        buy_index = (iodata.last_price > iodata.mid_price) & ~null_index
        sell_index = (iodata.last_price < iodata.mid_price) & ~null_index
        equal_index = (iodata.last_price == iodata.mid_price) & ~null_index

        iodata.loc[null_index, 'buy_sell'] = ''
        iodata.loc[null_index, 'buy_vol'] = 0
        iodata.loc[null_index, 'sell_vol'] = 0
        iodata.loc[null_index, 'transaction'] = False


        iodata.loc[buy_index, 'buy_sell'] = 'buy'
        iodata.loc[buy_index, 'buy_vol'] = iodata.loc[buy_index, 'volumm']
        iodata.loc[buy_index, 'sell_vol'] = 0
        iodata.loc[buy_index, 'transaction'] = True

        iodata.loc[sell_index, 'buy_sell'] = 'sell'
        iodata.loc[sell_index, 'buy_vol'] = 0
        iodata.loc[sell_index, 'sell_vol'] =  iodata.loc[sell_index, 'volumm']
        iodata.loc[sell_index, 'transaction'] = True

        temp_df = iodata[~null_index].copy()

        temp_df['bs_lag'] = temp_df.buy_sell.shift(1)
        temp_df['lp_lag'] = temp_df.last_price.shift(1)
        temp_df['transaction_lag'] = temp_df.transaction.shift(1)
        temp_df = temp_df.fillna(method='bfill') 
        temp_index = ~temp_df.transaction & temp_df.transaction_lag

        temp_sell_index = (temp_df.lp_lag > temp_df.last_price) & temp_index
        temp_df.loc[temp_sell_index, 'buy_sell'] = 'sell'
        temp_df.loc[temp_sell_index, 'sell_vol'] = temp_df.loc[temp_sell_index, 'volumm']
        temp_df.loc[temp_sell_index, 'buy_vol'] = 0
        temp_df.loc[temp_sell_index, 'transaction'] = True

        temp_buy_index = (temp_df.lp_lag < temp_df.last_price) & temp_index
        temp_df.loc[temp_buy_index, 'buy_sell'] = 'buy'
        temp_df.loc[temp_buy_index, 'sell_vol'] = 0
        temp_df.loc[temp_buy_index, 'buy_vol'] = temp_df.loc[temp_buy_index, 'volumm']
        temp_df.loc[temp_buy_index, 'transaction'] = True

        temp_equal_index = (temp_df.lp_lag == temp_df.last_price) & temp_index
        temp_df.loc[temp_equal_index, 'buy_sell'] = temp_df.loc[temp_equal_index, 'bs_lag']
        temp_df.loc[temp_equal_index, 'sell_vol'] = temp_df.loc[temp_equal_index, 'volumm'] 
        temp_df.loc[temp_equal_index, 'buy_vol'] = temp_df.loc[temp_equal_index, 'volumm'] 
        temp_df.loc[temp_equal_index, 'transaction'] = True



        a=temp_df[['instrument_id','minute','volumm','buy_vol','sell_vol']]
        b=iodata[null_index][['instrument_id','minute','volumm','buy_vol','sell_vol']]

        k=pd.concat([a,b])
        outdata=k.groupby([k.minute,k.instrument_id]).sum()
        outdata = outdata.reset_index()
        outdata['feat']=outdata.instrument_id.apply(lambda x: x[:7]+x[9:]) 
        outdata['date']=i
        outpath=self.outpathLR+i+'/'
        isexist=os.path.exists(outpath)
        if isexist==False:    
            os.makedirs(outpath)
        outdata.to_csv(outpath+ii)
        
    def LR_generates(self):
        for i in self.newfilelist[:]:
            subfilelist=os.listdir(self.datapath+i)
            iolist = self.iolist(subfilelist)
            try:         
                for ii in iolist: 
                    self.LR_detail(i,ii)
            except:
                pass

        
        
        

'''
这一步用于300股指这一步，用上面一部生成的data生成更多的相关factor
'''
        

        
        
    
class LeeReady(object):#datapath是上一步生成的文件的目录
    
    def __init__(self, datapath):
        self.datapath = datapath

    def AfterLR(self):  # 将每天的每分钟的所有的合约汇总成一个总的分钟级别的data
        # datapath=os.listdir('/home/researcher_k2/etfmmk2/Leeready/')
        # getMidData Lee Ready部分生成的文件路径

        PCR=[]
        for i in self.datapath:
            datapath2 = os.listdir(self.datapath+i)
            for j in datapath2:
                a=pd.read_csv(self.datapath+i+'/'+j)
                PCR.append(a)
        PCR=pd.concat(PCR, axis=0, ignore_index=True)
        return PCR


    def BuySellPCR(self,rollin):#生成每分钟买卖成交量的PCR，
        a= self.AfterLR().copy()
        a['c_p']=a.instrument_id.astype(str).str.contains("C")
        a['match']=a[['minute','feat','date']].apply(tuple,axis=1)
        Call=a[a.c_p==True]
        Put=a[a.c_p==False]
        print(len(Put))
        print(len(Call))
        b = pd.merge(Put,Call,on='match')

        K=pd.DataFrame(b.groupby(['date_x','minute_x'])[['buy_vol_x','sell_vol_x','buy_vol_y','sell_vol_y']].sum()).reset_index()
        K.columns=[['date','minute','buy_vol_put','sell_vol_put','buy_vol_call','sell_vol_call']]
        K['BUY_PCR']=K['buy_vol_call']/K['buy_vol_put']
        K['SELL_PCR']=K['sell_vol_call']/K['sell_vol_put']
        
        #这里生成以n分钟为滚动周期的PCR，rollin可以变化
        K['BUY_PCR_'+str(rollin)+'MIN']=K.buy_vol_call.rolling(rollin).sum()/K.buy_vol_put.rolling(rollin).sum()
        K['SELL_PCR_'+str(rollin)+'MIN']=K.sell_vol_call.rolling(rollin).sum()/K.sell_vol_put.rolling(rollin).sum()
        K = K[['date','minute','BUY_PCR','SELL_PCR','BUY_PCR_'+str(rollin)+'MIN','SELL_PCR_'+str(rollin)+'MIN']]
        return K
