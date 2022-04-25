#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd

def getDailyVol(close, span0=100):
    # daily vol, reindexed to cloes
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0>0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1 # daily returns
    df0 = df0.ewm(span=span0).std()
    return df0

def getVerticalBarriers(close, tEvents, numDays):
    t1 = close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=tEvents[:t1.shape[0]]) # NaNs at the end
    return t1

def group_bars(series, bar_idx):
    gg = series.groupby(bar_idx)
    df = pd.DataFrame()
    df['volume'] = gg['volume'].sum()
    if 'Dollar Volume' in series.columns:
        df['Dollar Volume'] = gg['Dollar Volume'].sum()
    df['open'] = gg['open'].first()
    df['low'] = gg['low'].first()
    df['high'] = gg['high'].last()
    df['close'] = gg['close'].last()
    if 'rPrices' in series.columns:
        df['rPrices'] = gg['rPrices'].last()
    #df['Instrument'] = gg['Instrument'].first()
    df['Time'] = gg.apply(lambda x:x.index[0])
    df['Num Ticks'] = gg.size()
    df = df.set_index(gg.apply(lambda x:x.index[0]))
    return df

def dollar_bars(series, bar_size=10000 * 3000):
    series = series.copy(deep=True)
    series['Dollar Volume'] = (series['volume'] * series['close'])
    series['Cum Dollar Volume'] = series['Dollar Volume'].cumsum()
    bar_idx = (series['Cum Dollar Volume'] / bar_size).round(0).astype(int).values
    return group_bars(series, bar_idx)

def cusum(gRaw, h):
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff()
    for i in diff.index[1:]:
        sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)

def mpPandasObj(func,pdObj,numThreads=24,mpBatches=1,linMols=True,**kargs):
    '''
    Parallelize jobs, return a dataframe or series
    + func: function to be parallelized. Returns a DataFrame
    + pdObj[0]: Name of argument used to pass the molecule
    + pdObj[1]: List of atoms that will be grouped into molecules
    + kwds: any other argument needed by func
    Example: df1=mpPandasObj(func,('molecule',df0.index),24,**kwds)
    '''
    import pandas as pd
    #if linMols:parts=linParts(len(argList[1]),numThreads*mpBatches)
    #else:parts=nestedParts(len(argList[1]),numThreads*mpBatches)
    if linMols:parts=linParts(len(pdObj[1]),numThreads*mpBatches)
    else:parts=nestedParts(len(pdObj[1]),numThreads*mpBatches)

    jobs=[]
    for i in range(1,len(parts)):
        job={pdObj[0]:pdObj[1][parts[i-1]:parts[i]],'func':func}
        job.update(kargs)
        jobs.append(job)
    if numThreads==1:out=processJobs_(jobs)
    else: out=processJobs(jobs,numThreads=numThreads)
    if isinstance(out[0],pd.DataFrame):df0=pd.DataFrame()
    elif isinstance(out[0],pd.Series):df0=pd.Series()
    else:return out
    for i in out:df0=df0.append(i)
    df0=df0.sort_index()
    return df0

def linParts(numAtoms,numThreads):
    # partition of atoms with a single loop
    parts=np.linspace(0,numAtoms,min(numThreads,numAtoms)+1)
    parts=np.ceil(parts).astype(int)
    return parts

def processJobs_(jobs):
    # Run jobs sequentially, for debugging
    out=[]
    for job in jobs:
        out_=expandCall(job)
        out.append(out_)
    return out

def expandCall(kargs):
    # Expand the arguments of a callback function, kargs['func']
    func=kargs['func']
    del kargs['func']
    out=func(**kargs)
    return out


def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads=1, t1=False, side=None):
    # 1) get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]
    # 2) get t1 (max holding period)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
    # 3) form events object, apply stop loss on t1
    if side is None:
        side_, ptSl_ = pd.Series(1.0, index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        side_, ptSl_ = side.loc[trgt.index], ptSl[:2]
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=[('trgt', 'close')])
    df0 = mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule', events.index), numThreads=numThreads, close=close, events=events, ptSl=ptSl_)
    events['t1'] = df0.dropna(how='all').min(axis=1) # pd.min ignores NaN
    if side is None:
        events = events.drop('side', axis=1)

    # store for later
    events['pt'] = ptSl[0]
    events['sl'] = ptSl[1]

    return events
    
def applyPtSlOnT1(close, events, ptSl, molecule):
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)

    if ptSl[0] > 0:
        pt = ptSl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index) # NaNs

    if ptSl[1] > 0:
        sl = - ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index) # 'mo NaNs
    
    for l, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        if l==0: continue
        df0 = close[l:t1] # path prices
        df0 = (df0 / close[l] - 1) * events_.at[l, 'side'] # path returns
        out.loc[l, 'sl'] = df0[df0<sl[l]].index.min() # earliest stop loss
        out.loc[l, 'pt'] = df0[df0>pt[l]].index.min() # earliest profit take
    return out

def getBins(events, close):
    # 1) prices aligned with events
    events_ = events.dropna(subset=[('t1',0)])
    px = events_.index.union(events_[('t1',0)].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = (px.loc[events_[('t1',0)].values].values / px.loc[events_.index]) - 1
    out['bin'] = np.sign(out['ret'])
    return out

def getWeights(d,size):
    # thres>0 drops insignificant weights
    w=[1.]
    for k in range(1,size):
        w_=-w[-1]/k*(d-k+1)
        w.append(w_) 
    w=np.array(w[::-1]).reshape(-1,1) 
    return w

def plotWeights(dRange,nPlots,size):
    w=pd.DataFrame()
    for d in np.linspace(dRange[0],dRange[1],nPlots):
        w_=getWeights(d,size=size) 
        w_=pd.DataFrame(w_,index=range(w_.shape[0])[::-1],columns=[d]) 
        w=w.join(w_,how='outer')
    ax=w.plot()
    ax.legend(loc='upper left');mpl.show() 
    return

def fracDiff(series,d,thres=.01): 
    '''
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1]. 
    '''
    #1) Compute weights for the longest series 
    w=getWeights(d,series.shape[0])
    #2) Determine initial calcs to be skipped based on weight-loss threshold 
    w_=np.cumsum(abs(w))
    w_/=w_[-1]
    skip=w_[w_>thres].shape[0]
    #3) Apply weights to values
    df={}
    for name in series.columns:
        seriesF,df_=series[[name]].fillna(method='ffill').dropna(),pd.Series() 
        for iloc in range(skip,seriesF.shape[0]):
            loc=seriesF.index[iloc]
            if not np.isfinite(series.loc[loc,name]):
                continue # exclude NAs 
            df_[loc]=np.dot(w[-(iloc+1):,:].T,seriesF.loc[:loc])[0,0]
        df[name]=df_.copy(deep=True) 
    df=pd.concat(df,axis=1)
    return df

def getWeights_FFD(d, thres):
    w, k = [1.0], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def fracDiff_FFD(series,d,thres=1e-5): 
    '''
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1]. 
        '''
    #1) Compute weights for the longest series
    w=getWeights_FFD(d,thres)
    width=len(w)-1
    #2) Apply weights to values
    df={}
    for name in series.columns: 
        seriesF,df_=series[[name]].fillna(method='ffill').dropna(),pd.Series() 
        for iloc1 in range(width,seriesF.shape[0]):
            loc0,loc1=seriesF.index[iloc1-width],seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]):continue # exclude NAs 
            df_[loc1]=np.dot(w.T,seriesF.loc[loc0:loc1])[0,0]
        df[name]=df_.copy(deep=True) 
    df=pd.concat(df,axis=1)
    return df

def dropLabels(events, mitPct=0.05):
    # apply weights, drop labels with insufficient examples
    while True:
        df0 = events['bin'].value_counts(normalize=True)[::-1]
        if df0.min() > mitPct or df0.shape[0] < 3:
            break
        print("Dropped label", df0.argmin(), df0.min())
        events = events[events['bin'] != df0.argmin()]
    return events


def barrier_touched(out_df, events):
    # We'll graciously use the barrier_touched method from
    # https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/labeling/labeling.py#L164
    store = []
    for date_time, values in out_df.iterrows():
        ret = values['ret']
        target = values['trgt']

        pt_level_reached = ret > target * events.loc[date_time, 'pt']
        sl_level_reached = ret < -target * events.loc[date_time, 'sl']

        if ret > 0.0 and pt_level_reached.item():
            # Top barrier reached
            store.append(1)
        elif ret < 0.0 and sl_level_reached.item():
            # Bottom barrier reached
            store.append(-1)
        else:
            # Vertical barrier reached
            store.append(0)

    # Save to 'bin' column and return
    out_df['bin'] = store
    return out_df

def getMetaBins(events, close):
    '''
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    Case 1: ('side' not in events): bin in (-1, 1) <- label by price action
    Case 2: ('side' in events): bin in (0, 1) <- label by pnl (meta-labeling)
    '''
    # 1) prices aligned with events
    events_ = events.dropna(subset=[('t1',0)])
    px = events_.index.union(events_[('t1',0)].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    # 2) create out object
    out = pd.DataFrame(index=events_.index)

    out['ret'] = px.loc[events_[('t1',0)].values].values / px.loc[events_.index] - 1
    if 'side' in events_:
        out['ret'] *= events_['side']  # meta-labeling

    out['trgt'] = events_['trgt']
    out = barrier_touched(out, events)

    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0
        
    if 'side' in events_:
        out['side'] = events['side']
    return out


def mpNumCoEvents(closeIdx, t1, molecule):
    '''
    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed
    +molecule[-1] is the date of the last event on which the weight will be computed
    Any event that starts before t1[molecule].max() impacts the count.
    '''
    # 1) find events that span the period [molecule[0], molecule[-1]]
    t1 = t1.fillna(closeIdx[-1]) # unclosed events still must impact other weights
    t1 = t1[t1 >= molecule[0]] # events that end at or after molecule[0]
    t1 = t1.loc[:t1[molecule].max()] # events that start at or before t1[molecule].max()
    # 2) count events spanning a bar
    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=closeIdx[iloc[0]:iloc[1]+1])
    for tIn, tOut in t1.iteritems():
        if tIn==0: continue
        count.loc[tIn:tOut] += 1.0
    return count.loc[molecule[0]:t1[molecule].max()]

def mpSampleTW(t1, numCoEvents, molecule):
    # Derive average uniqueness over the event's lifespan
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn] = (1.0 / numCoEvents.loc[tIn:tOut]).mean()
    return wght

def getIndMatrix(barIx,t1):
    # Get indicator matrix
    indM=pd.DataFrame(0,index=barIx,columns=range(t1.shape[0]))
    for i,(t0,t1) in enumerate(t1.iteritems()):
        indM.loc[t0:t1,i]=1. 
    return indM

def seqBootstrap(indM,sLength=None):
# Generate a sample via sequential bootstrap 
    if sLength is None:sLength=indM.shape[1] 
    phi=[]
    while len(phi)<sLength:
        avgU=pd.Series() 
        for i in indM:
            indM_=indM[phi+[i]] # reduce indM
            avgU.loc[i]=getAvgUniqueness(indM_).iloc[-1] 
        prob=avgU/avgU.sum() # draw prob 
        phi+=[np.random.choice(indM.columns,p=prob)]
    return phi

def mpSampleW(t1,numCoEvents,close,molecule):
    # Derive sample weight by return attribution 
    ret=np.log(close).diff() # log-returns, so that they are additive 
    wght=pd.Series(index=molecule)
    for tIn,tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn]=(ret.loc[tIn:tOut]/numCoEvents.loc[tIn:tOut]).sum() 
    return wght.abs()

def getTimeDecay(tW,clfLastW=1.):
    # apply piecewise-linear decay to observed uniqueness (tW)
    # newest observation gets weight=1, oldest observation gets weight=clfLastW 
    clfW=tW.sort_index().cumsum()
    if clfLastW>=0:slope=(1.-clfLastW)/clfW.iloc[-1] 
    else:slope=1./((clfLastW+1)*clfW.iloc[-1])
    const=1.-slope*clfW.iloc[-1]
    clfW=const+slope*clfW
    clfW[clfW<0]=0
    print (const,slope)
    return clfW


def mpNumCoEvents(closeIdx, t1, molecule):
    '''
    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed
    +molecule[-1] is the date of the last event on which the weight will be computed
    Any event that starts before t1[molecule].max() impacts the count.
    '''
    # 1) find events that span the period [molecule[0], molecule[-1]]
    t1 = t1.fillna(closeIdx[-1]) # unclosed events still must impact other weights
    t1 = t1[t1 >= molecule[0]] # events that end at or after molecule[0]
    t1 = t1.loc[:t1[molecule].max()] # events that start at or before t1[molecule].max()
    # 2) count events spanning a bar
    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=closeIdx[iloc[0]:iloc[1]+1])
    for tIn, tOut in t1.iteritems():
        if tIn==0: continue
        count.loc[tIn:tOut] += 1.0
    return count.loc[molecule[0]:t1[molecule].max()]

def mpSampleTW(t1, numCoEvents, molecule):
    # Derive average uniqueness over the event's lifespan
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn] = (1.0 / numCoEvents.loc[tIn:tOut]).mean()
    return wght

def getIndMatrix(barIx,t1):
    # Get indicator matrix
    indM=pd.DataFrame(0,index=barIx,columns=range(t1.shape[0]))
    for i,(t0,t1) in enumerate(t1.iteritems()):
        indM.loc[t0:t1,i]=1. 
    return indM

def seqBootstrap(indM,sLength=None):
# Generate a sample via sequential bootstrap 
    if sLength is None:sLength=indM.shape[1] 
    phi=[]
    while len(phi)<sLength:
        avgU=pd.Series() 
        for i in indM:
            indM_=indM[phi+[i]] # reduce indM
            avgU.loc[i]=getAvgUniqueness(indM_).iloc[-1] 
        prob=avgU/avgU.sum() # draw prob 
        phi+=[np.random.choice(indM.columns,p=prob)]
    return phi

def mpSampleW(t1,numCoEvents,close,molecule):
    # Derive sample weight by return attribution 
    ret=np.log(close).diff() # log-returns, so that they are additive 
    wght=pd.Series(index=molecule)
    for tIn,tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn]=(ret.loc[tIn:tOut]/numCoEvents.loc[tIn:tOut]).sum() 
    return wght.abs()

def getTimeDecay(tW,clfLastW=1.):
    # apply piecewise-linear decay to observed uniqueness (tW)
    # newest observation gets weight=1, oldest observation gets weight=clfLastW 
    clfW=tW.sort_index().cumsum()
    if clfLastW>=0:slope=(1.-clfLastW)/clfW.iloc[-1] 
    else:slope=1./((clfLastW+1)*clfW.iloc[-1])
    const=1.-slope*clfW.iloc[-1]
    clfW=const+slope*clfW
    clfW[clfW<0]=0
    print (const,slope)
    return clfW

def getAvgUniqueness(indM):
    # Average uniqueness from indicator matrix 
    c=indM.sum(axis=1) # concurrency 
    u=indM.div(c,axis=0) # uniqueness 
    avgU=u[u>0].mean() # average uniqueness 
    return avgU