from metrics import * 
import pandas as pd
from shrp_calmar import * 
import numpy as np
from scipy import stats
from BTtools import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import KmeansTool as KT

def loopforbest (factor2,factor_list,thresrange,movlenrange,freqrange,tname,rname):
    
    K,IC,Signal_Ts,factor_Sum,Factor_name,Threshold,step=[],[],[],[],[],[],[]
    falist1=factor_list
    movlen = movlenrange
    for S in freqrange:#每S分钟进行一次交易判断
        for i in falist1:#对于每一个factor，要进行threshold 和 movinglength的穷举
            try:
                for n in thresrange:
                    for m in movlenrange:
                        A = list(range(0,len(factor2),S))
                        factor21 = factor2.reset_index().iloc[A].set_index(tname)
                        skew_threshold_high, skew_threshold_low = get_bolling(factor21[i],n,m)
                        TS = tuple([n,m])
                        Threshold.append(TS)
                        rst = backtest(pd.to_datetime(list(map(str, factor21.index))), factor21[rname].values, factor21[i].values, skew_threshold_high, skew_threshold_low, 1, 0)
                        k = result_to_str(rst[0][0], simple_key_list).split('|')
                        factor_Sum.append(k)#对于每一个 factor 生成一个movinglength 一个threshold 组合然后记录其各种表现
                        IC.append(get_IC2(factor21, factor21[i])[rname])
                        Signal_Ts.append(rst[1])
                        Factor_name.append(i)
                        step.append(S)
            except:
                pass
            


    Threshold = pd.DataFrame(Threshold)
    Threshold.columns=['threshold','movinglength']
    step = pd.DataFrame(step)
    step.columns=['step']
    factor_Sum = pd.DataFrame(factor_Sum)
    factor_Sum.columns = simple_key_list
    IC = pd.DataFrame(IC)
    IC.columns = ['IC']
    Signal_Ts = pd.DataFrame(Signal_Ts)
    Signal_Ts.columns = ['signaltime']
    Factor_name = pd.DataFrame(Factor_name)
    Factor_name.columns = ['Factorname']
    factor_Sum.insert(6,'IC',IC['IC'].values)
    factor_Sum.insert(7,'signaltime',Signal_Ts['signaltime'].values)
    factor_Sum.insert(8,'threshold',Threshold['threshold'].values)
    factor_Sum.insert(9,'movinglength',Threshold['movinglength'].values)
    factor_Sum = factor_Sum.applymap(lambda x: round(float(x),3))
    factor_Sum.insert(10,'Factorname',Factor_name['Factorname'].values)
    factor_Sum.insert(11,'steps',step['step'].values)
    #以表格形式保存
    #factor_Sum.drop(0).drop(22).drop(10)   
    return factor_Sum



def get_groupnumber(numofgroup,data_path):#这一个函数为了生成一个list来判断大致可以把这个数据分为几个族群
    data = pd.read_csv(data_path)
    dataKM = data.loc[data.sharpe_ratio>0][['sharpe_ratio','max_drawdown','signaltimes']]#首先过滤掉指标为负数的结果
    list_lost = []
    for k in numofgroup:# 
        min_loss = 10000
        min_loss_centroids = np.array([])
        min_loss_clusterData = np.array([])
        for i in range(50):#牛顿趋近法来记录生成不同组数最后与族群原点的欧几里和距离作为loss
            centroids, clusterData = KT.kmeans(dataKM.values, k)  
            loss = sum(clusterData[:,1])/data.shape[0]
            if loss < min_loss:
                min_loss = loss
                min_loss_centroids = centroids
                min_loss_clusterData = clusterData
        list_lost.append(min_loss)
    
    plt.plot(numofgroup,list_lost)
    plt.xlabel('k')
    plt.ylabel('loss')
    plt.show()
    print(list_lost)
    return list_lost#判断通过这个list的衰减点
    

def Km_Optm(num, path): #num是最终确定的族群数目
    data = pd.read_csv(path)
    dataKM = data.loc[data.sharpe_ratio>0][['sharpe_ratio','max_drawdown','signaltimes']]#用KMEANS 以这三个指标（或者更多来进行分组）
    model = KMeans(n_clusters=num)
    model.fit(dataKM)
    centers = model.cluster_centers_
    result = model.predict(dataKM)
    K=data.loc[data.sharpe_ratio>0]
    K=K.reset_index()
    resultSum = pd.concat([K,pd.DataFrame(result)],axis = 1)
    return resultSum #最后生成的这个结果相当于上面Loopforbest函数的输出然后加上组数的标记（1，2，3.。。）