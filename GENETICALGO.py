#coding = utf-8
from metrics import * 
from BTtools import *
import pandas as pd
from shrp_calmar import * 
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from math import *
from mpl_toolkits.mplot3d import Axes3D


class ga_opt(object):
    '''
    值直接call result（）即可生成最优参数组合
    '''

    def __init__(self, kind,factors,returns,Crossover_rate,
                 Mutation_rate,maxnum,maxx,maxy,N): 
        '''
        初始化一个属性r（不要忘记self参数，他是类下面所有方法必须的参数）
        表示给我们将要创建的实例赋予属性r赋值
        :param kind:   kind为一个基因pool我们需要多少条基因
        :param factors: 因子的数据
        :param return：收益率的数据
        :param DNA: 用来记录每次迭代基因库的变化
        :param _FDNA: 用来记录每次迭代的适应度数值
        :param Crossover rate：基因交叉的概率
        :param Mutation_rate：基因变异的概率
        :param N：总的迭代次数
        '''
        
        self.kind = kind   
        self.factors = factors
        self.returns = returns
        self.DNA = []
        self.NDNA = [0]*kind
        self.NNDNA = [0]*kind
        self._FDNA = [0]*kind  
        self.Crossover_rate = Crossover_rate
        self.Mutation_rate = Mutation_rate
        self.N = N
        self.maxnum = maxnum
        self.maxx = maxx
        self.maxy = maxy# 通常初始设置为0
# kind = 6 #种群密度
# DNA = []
# NDNA = [0]*kind
# NNDNA = [0]*kind
# _FDNA = [0]*kind
# Crossover_rate = 0.40 #交叉概率
# Mutation_rate = 0.08 #变异概率
# N = 50
# maxnum = 0
# maxx = 0
# maxy = 0
    def fitness(self, x1, x2):  # 这个是我们的目标函数以优化sharp，BPS，anuual return为目标
        skew_threshold_high, skew_threshold_low = get_bolling(self.factors,x1, x2)
        rst = BTGA(pd.to_datetime(list(map(str, self.factors.index))), self.returns, self.factors, skew_threshold_high, skew_threshold_low, 1, 0)
        A,S,B = rst[0],rst[1],rst[2]    
        K = 1/((1/S)+(2/B)+(3/A))
        return K

    def tobin(self,num): # 将二进制转为数字
        nnum = 0
        l = len(num)
        for i in range(l-1, -1, -1):
            if num[l-1-i] == 1:
                nnum += (2**i)
        return nnum/100

    def update(self):#每次进行selection， cross，mute之后要更新这个基因库
        _sum = 0
        for i in range(self.kind):
            xx1 = self.DNA[i][:9]
            xx2 = self.DNA[i][9:]
            num1 = self.tobin(xx1)
            num2 = int(self.tobin(xx2)+1)*20
            self.NDNA[i] = self.fitness(num1, num2)
            if self.NDNA[i] > self.maxnum:
                self.maxnum = self.NDNA[i]
                self.maxx = num1
                self.maxy = num2
        for i in range(self.kind):
            self.NNDNA[i] = (self.NDNA[i] - min(self.NDNA)-abs(min(self.NDNA)*0.1))
            _sum += self.NNDNA[i]
        __sum = 0
        for i in range(self.kind):
            __sum += self.NNDNA[i]
            self._FDNA[i] = __sum/_sum
        #print(NDNA)
        #print(_FDNA)

    def fix(self, ss):#将两个数从一串二进制序列中分开
        s1 = ss[:9]
        s2 = ss[9:]
        num1 = round(self.tobin(s1)*100)
        num2 = round(self.tobin(s2)*100)
        # print('num1 = {}, num2 = {}'.format(num1, num2))
        #     num1 %= 1490000
        #     num2 %= 150000
        # print('num1n = {}, num2n = {}'.format(num1, num2))
        n1 = str(bin(num1))
        n2 = str(bin(num2))
        n1 = n1[2:]
        n2 = n2[2:]
        if len(n1) < 9:#这里的9和11是为了达成小数点后两位的目的，可以根据具体任务变化
            nnn = 9 - len(n1)
            for tt in range(nnn):
                n1 = '0' + n1
        if len(n2) < 11:
            nnn = 11 - len(n2)
            for tt in range(nnn):
                n2 = '0' + n2
        for i in range(len(s1)):
            s1[i] = int(n1[i])
        for i in range(len(s2)):
            s2[i] = int(n2[i])
        return s1+s2

    def init(self): #初始化我们的基因库
        for i in range(self.kind):
            s = [0]*20
            for j in range(20):#单个的一条基因需要20个二进制bit 可根据需要来变化
                s[j] = random.randint(0, 1)
            s = self.fix(s)
            self.DNA.append(s)
        self.update()

    def selection(self):#随机选基因pool中的某个进行cross or mute
        DNA_ = []
        for i in range(self.kind):
            num = random.random()
            for j in range(self.kind):
                if j == 0 and num <= self._FDNA[j]:
                    DNA_.append(self.DNA[j])
                elif num <= self._FDNA[j] and num > self._FDNA[j-1]:
                    DNA_.append(self.DNA[j])
        for i in range(self.kind):
            self.DNA[i] = DNA_[i]
        self.update()

    def cross(self):#单个基因以一定的概率进行交叉
        a1 = random.randint(0, self.kind-1)
        a2 = random.randint(0, self.kind-1)
        point = random.randint(0, 20)
        ans1 = self.DNA[a1][:point]
        ans2 = self.DNA[a1][point:]
        ans3 = self.DNA[a2][:point]
        ans4 = self.DNA[a2][point:]
        _ans1 = ans1+ans4
        _ans2 = ans2+ans3
        _ans1 = self.fix(_ans1)
        _ans2 = self.fix(_ans2)
        self.DNA[a1] = _ans1
        self.DNA[a2] = _ans2
        self.update()

    def variation(self):#模仿基因以某小概率变异
        for i in range(int(self.Mutation_rate*20*self.kind)):
            ind = random.randint(0, self.kind-1)
            ind1 = random.randint(0, 19)
            self.DNA[ind][ind1] ^= 1
        for i in range(self.kind):
            self.DNA[i] = self.fix(self.DNA[i])
        self.update()
        
    def result(self): #对于我们的打算多数task直接call这个reesult函数即可以生成最优化结果
        
        self.init()
        _ans = []
        ans = []
        for i in tqdm(range(self.N)):
            self.selection()
            crand = random.random()
            mrand = random.random()
            if crand <= self.Crossover_rate:
                self.cross()
            self.variation()
            #print('solutions are', NDNA, end=' ')
            #print('The optimal solution now is', max(NDNA))
            _ans.append(max(self.NDNA))
            ans.append(max(_ans))
            try:
                if (i-int(0.5*self.N))>0:
                    if ans[i]==ans[i-int(0.5*self.N)]:
                        break
            except: 
                pass
        print('The optimal solution finally is', max(_ans))
        print('x = {:.5f}, y = {:.5f} '.format(self.maxx, self.maxy))

        



