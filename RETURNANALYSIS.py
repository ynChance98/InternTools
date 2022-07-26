from metrics import * 
import pandas as pd
from shrp_calmar import * 
import numpy as np
from scipy import stats
from datetime import datetime   
import statsmodels as sm
from sklearn.linear_model import LinearRegression
import statsmodels.regression.linear_model as sm


class F_retAn(object):
    '''
    需要数据清洗后的数据，数据中需因子值和收益率
    '''
    
    def __init__(self, factor_data): 
        '''
        初始化一个属性r（不要忘记self参数，他是类下面所有方法必须的参数）
        表示给我们将要创建的实例赋予属性r赋值
        '''
        self.factor_data = factor_data
    
    def correlation(self):
        return pd.DataFrame(self.factor_data.corr())
    
    def LinearRegres(self,x1,x2):#x1因子，y1收益率,不能是series必须是values
        self.factor_data=self.factor_data.fillna(method='pad')
        a = np.hstack([np.where(np.isinf(x1))[0],np.where(np.isinf(x2))[0]])
        a=a.flatten()
        x1=np.delete(x1,a)
        x2=np.delete(x2,a)
        x1=x1.reshape((-1, 1))
        x2=x2.reshape((-1, 1))
        #以上数据清洗去掉inf和空值
        model1 = sm.OLS(endog=x1, exog=x2).fit()
        return model1.summary()
