#!/usr/bin/env python
# coding: utf-8

# In[30]:


class alpha_vantage():
    
    def save(self,file,name,path='/Users/abhijitdeshpande/Documents/Project Files/Data with Dates/'):
        '''file: Data want to be saved in CSV format 
           name: name of file to be named
           path: directory where you want to save your file'''
        import pandas as pd
        file.to_csv(path+name+'.csv')
            
    def retrive_data(self,company_name, interval,period=200):
        import alpha_vantage, time
        api_key = 'SXJIW6MO6ZIEYB0N'
        from alpha_vantage.techindicators import TechIndicators
        from alpha_vantage.timeseries import TimeSeries

        ts = TimeSeries(key=api_key, output_format='pandas')
        ta = TechIndicators(key=api_key, output_format='pandas')

        def technical_features(technical_feature,company=company_name, interval=interval, period=period, series_type='Close'):
            data, meta_data = technical_feature(company, interval=interval, time_period= period, series_type=series_type)
            return data

        def other_features(technical_feature,company=company_name, interval=interval):
            data, meta_data = technical_feature(company, interval=interval)
            return data

        adx = other_features(ta.get_adx,company_name,interval)
        cci = other_features(ta.get_cci,company_name,interval)
        arron = technical_features(ta.get_aroon,company_name,interval)
        ad = other_features(ta.get_ad,company_name,interval)
        obv = other_features(ta.get_obv,company_name,interval)

        time.sleep(60)

        bbands = other_features(ta.get_bbands,company_name,interval)
        wma = other_features(ta.get_wma,company_name,interval)
        dema = other_features(ta.get_dema,company_name,interval)
        tema = other_features(ta.get_tema,company_name,interval)
        trima = other_features(ta.get_trima,company_name,interval)

        time.sleep(60)

        sma = technical_features(ta.get_sma,company_name,interval)
        ema = technical_features(ta.get_ema,company_name,interval)
        macd = other_features(ta.get_macd,company_name,interval)
        stoch = other_features(ta.get_stoch,company_name,interval)
        rsi = other_features(ta.get_rsi,company_name,interval)

        time.sleep(60)

        kama = other_features(ta.get_kama,company_name,interval)
        mama = other_features(ta.get_mama,company_name,interval)
        t3 = other_features(ta.get_t3,company_name,interval)
        macdext = other_features(ta.get_macdext,company_name,interval)
        stochf = other_features(ta.get_stochf,company_name,interval)

        time.sleep(60)

        stochrsi = other_features(ta.get_stochrsi,company_name,interval)
        willr = other_features(ta.get_willr,company_name,interval)
        adxr = other_features(ta.get_adxr,company_name,interval)
        apo = other_features(ta.get_apo,company_name,interval)
        ppo = other_features(ta.get_ppo,company_name,interval)

        time.sleep(60)

        mom = other_features(ta.get_mom,company_name,interval)
        bop = other_features(ta.get_bop,company_name,interval)
        cmo = other_features(ta.get_cmo,company_name,interval)
        roc = other_features(ta.get_roc,company_name,interval)
        rocr = other_features(ta.get_rocr,company_name,interval)

        time.sleep(60)

        aroonosc = other_features(ta.get_aroonosc,company_name,interval)
        mfi = other_features(ta.get_mfi,company_name,interval)
        trix = other_features(ta.get_trix,company_name,interval)
        ultosc = other_features(ta.get_ultosc,company_name,interval)
        dx = other_features(ta.get_dx,company_name,interval)

        time.sleep(60)

        minus_di = other_features(ta.get_minus_di,company_name,interval)
        plus_di = other_features(ta.get_plus_di,company_name,interval)
        minus_dm = other_features(ta.get_minus_dm,company_name,interval)
        plus_dm = other_features(ta.get_plus_dm,company_name,interval)
        midpoint = other_features(ta.get_midpoint,company_name,interval)

        time.sleep(60)

        midprice = other_features(ta.get_midprice,company_name,interval)
        sar = other_features(ta.get_sar,company_name,interval)
        trange = other_features(ta.get_trange,company_name,interval)
        atr = other_features(ta.get_atr,company_name,interval)
        natr = other_features(ta.get_natr,company_name,interval)

        time.sleep(60)

        ht_trendline = other_features(ta.get_ht_trendline,company_name,interval)
        ht_sine = other_features(ta.get_ht_sine,company_name,interval)
        ht_trendmode = other_features(ta.get_ht_trendmode,company_name,interval)
        ht_dcperiod = other_features(ta.get_ht_dcperiod,company_name,interval)
        ht_dcphase = other_features(ta.get_ht_dcphase,company_name,interval)

        time.sleep(60)

        ht_phasor = other_features(ta.get_ht_phasor,company_name,interval)
        if interval == 'daily':
            adj, meta = ts.get_daily(company_name,outputsize='full')
            
        else:
            adj, meta = ts.get_intraday(company_name,outputsize='full',interval=interval)
        
        data = adj[['1. open', '2. high', '3. low', '4. close','5. volume']][::-1]
        data.rename(columns={'1. open':'Open', '2. high':'High', '3. low':'Low','4. close':'Close',                                               '5. volume':'Volume'},inplace=True)
                                                               

        feature = pd.concat([sma,ema,macd,stoch,rsi,adx,cci,
                         arron,ad,obv,bbands,wma,dema,tema,trima,kama,
                         mama,t3,macdext,stochf,stochrsi,willr,adxr,apo,
                         ppo,mom,bop,cmo,roc,rocr,aroonosc,mfi,trix,ultosc,
                         dx,minus_di,plus_di,minus_dm,plus_dm,midpoint,
                         midprice,sar,trange,atr,natr,ht_trendline,
                         ht_sine,ht_trendmode,ht_dcperiod,ht_dcphase,
                         ht_phasor],axis=1,join='inner')

        final_data = pd.concat([feature,data],axis=1).dropna()
        
        return final_data

