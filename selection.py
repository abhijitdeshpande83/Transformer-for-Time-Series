#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import talib as ta
from talib import abstract
import warnings
warnings.filterwarnings('ignore')


# In[3]:


class generate_labels():
    
    def __init__(self,data,holding_period=30,profit=4,risk=1):
        
        self.data = data
        data.index = data.time
        self.holding_period = holding_period
        self.profit = profit
        self.risk = risk
        self.upper_lower_multipliers = [self.profit, self.risk]


    def get_Daily_Volatility(self,span0=20):
        # simple percentage returns
        df0=self.data.close.pct_change()
        # 20 days, a month EWM's std as boundary
        df0=df0.ewm(span=span0).std()
        df0.dropna(inplace=True)
        return df0

    def get_barriers(self):
        #create a container
        df0 = self.get_Daily_Volatility()
        
        prices = self.data.close[df0.index]

        barriers = pd.DataFrame(columns=['days_passed', 
                  'price', 'vert_barrier', \
                  'top_barrier', 'bottom_barrier'], \
                   index = df0.index)
        for day, vol in df0.iteritems():
            days_passed = len(df0.loc[df0.index[0] : day])
            #set the vertical barrier 
            if ((days_passed + self.holding_period) < len(df0.index) and self.holding_period != 0):
                vert_barrier = df0.index[
                                    days_passed + self.holding_period]
            else:
                vert_barrier = np.nan
            #set the top barrier
            if self.upper_lower_multipliers[0] > 0:
                top_barrier = prices.loc[day] + prices.loc[day] *self.upper_lower_multipliers[0] * vol
            else:
                #set it to NaNs
                top_barrier = pd.Series(index=prices.index)
            #set the bottom barrier
            if self.upper_lower_multipliers[1] > 0:
                bottom_barrier = prices.loc[day] - prices.loc[day] *self.upper_lower_multipliers[1] * vol
            else: 
                #set it to NaNs
                bottom_barrier = pd.Series(index=prices.index)
                
            barriers.loc[day, ['days_passed', 'price', 
                'vert_barrier','top_barrier', 'bottom_barrier']] = \
                 days_passed, prices.loc[day], vert_barrier, \
                 top_barrier, bottom_barrier
            
        return barriers
        
    def get_labels(self):
        
        barriers = self.get_barriers()
 
        barriers['out'] = np.nan

        for i in range(len(barriers.index)):
        
            start = barriers.index[i]
            end = barriers.vert_barrier[i]
            if pd.notna(end):
                # assign the initial and final price
                price_initial = barriers.price[start]
                price_final = barriers.price[end]
                # assign the top and bottom barriers
                top_barrier = barriers.top_barrier[i]
                bottom_barrier = barriers.bottom_barrier[i]
                #set the profit taking and stop loss conditons
                condition_pt = (barriers.price[start: end] >=top_barrier).any()
                condition_sl = (barriers.price[start: end] <=bottom_barrier).any()
                #assign the labels

                if condition_pt:
                    barriers['out'][i] = 1
                    
                elif condition_sl: 
                    barriers['out'][i] = -1    
                else: 
                    barriers['out'][i] = max(
                              [(price_final - price_initial)/ 
                               (top_barrier - price_initial), \
                               (price_final - price_initial)/ \
                               (price_initial - bottom_barrier)],\
                                key=abs)
                    #barriers['out'][i] = 0
                    
        self.barriers = barriers

        return pd.concat([self.data,barriers[['out']]],axis=1).drop('time',axis=1).dropna()
    
    def plot(self,x=10,y=8):
        """x: length of plot
           y: width of plot"""
        
        point = random.choice(range(len(self.barriers)))
        print('From: ',self.data.index[point],' To:',self.data.index[point+self.holding_period])
        fig,ax = plt.subplots(figsize=(x,y))
        ax.set(title=self.barriers.out.iloc[point],
               xlabel='date', ylabel='price')
        ax.plot(self.barriers.price[point: point+self.holding_period])
        start = self.barriers.index[point]
        end = self.barriers.vert_barrier[point]
        upper_barrier = self.barriers.top_barrier[point]
        lower_barrier = self.barriers.bottom_barrier[point]
        ax.plot([start, end], [upper_barrier, upper_barrier], 'r--');
        ax.plot([start, end], [lower_barrier, lower_barrier], 'r--');
        ax.plot([start, end], [(lower_barrier + upper_barrier)*0.5, (lower_barrier + upper_barrier)*0.5], 'r--');
        ax.plot([start, start], [lower_barrier, upper_barrier], 'r-');
        ax.plot([end, end], [lower_barrier, upper_barrier], 'r-');
        
def save(file, name, path='/Users/abhijitdeshpande/Documents/Project Files/Full Hourly Data/',index=False):
        file.to_csv(path+name+'.csv',index=index)
        

def load_data(name,path='/Users/abhijitdeshpande/Downloads/DJI65_2Y60min/'):
    return pd.read_csv(path+name+'.csv')


class TechnicalIndicator():
    
    def __init__(self):
        self.indicators = ['MA','SMA','EMA','MACD','RSI','PPO','SAR','STDDEV','CCI','OBV',
                      'STOCHRSI','AROON','SAR','ADX','BBANDS','WILLR','AROONOSC','AD',\
                        'MFI','CORREL','ATR','MOM','MINUS_DM','MINUS_DI','DX',\
                        'ROC','TRIX','ULTOSC','BOP','TSF','LINEARREG','LINEARREG_ANGLE',\
                        'LINEARREG_INTERCEPT','LINEARREG_SLOPE','BETA']

    def add(self,feature):
        self.indicators.append(feature)

    def __call__(self,data):

        for i in self.indicators:
            try:
                data[i] = abstract.Function(i)(data)
            except:
                data[list(abstract.Function(i)(data).columns)] = abstract.Function(i)(data)
        return data.dropna()

