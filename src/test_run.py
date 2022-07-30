#!/usr/bin/env python
# coding: utf-8

# # test_file

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# In[2]:


os.getcwd()


# In[3]:


os.chdir('/Users/abhijitdeshpande/Downloads/DJI65_2Y60min')


# In[4]:


file = [x.split('_')[0] for x in os.listdir()]


# In[ ]:





# In[5]:


name = file[np.random.randint(len(file))]


# In[3403]:


name = 'KSU'


# In[3404]:


data = pd.read_csv('/Users/abhijitdeshpande/Downloads/DJI65_2Y60min/'+name+'_2Y60min.csv')


# In[3405]:


data.time = pd.to_datetime(data['time'])
data.set_index('time',inplace=True)


# In[3406]:


data.close.plot()


# In[3407]:


#data = data[:7690]


# In[3408]:


os.chdir('/Users/abhijitdeshpande/Documents/Project Files/Py files')


# In[3409]:


from utils import *


# In[3410]:


name


# In[3411]:


from statsmodels.tsa.stattools import adfuller


# In[3412]:


data.volume.mean()


# In[ ]:





# In[ ]:





# In[3413]:


d_bars = dollar_bars(data,3e6)
daily_volatility = getDailyVol(d_bars[['close']])
daily_vol_mean = daily_volatility.mean().item()


# In[ ]:





# In[3414]:


sampled_d_bars = cusum(d_bars['close'],daily_vol_mean)
close = d_bars['close']
tEvents = sampled_d_bars
numDays = 3
t1 = getVerticalBarriers(d_bars,tEvents,numDays)


# In[3415]:


len(d_bars)


# In[3416]:


events = getEvents(d_bars['close'], tEvents=tEvents[8:], ptSl=[2,2], t1=t1,                   numThreads=1, trgt=daily_volatility, minRet=0.01)


# In[3417]:


events


# In[ ]:





# In[ ]:





# In[3418]:


plt.figure(figsize=(16,12))
#d_bars.close.plot()
data.close.plot(color='black')
plt.scatter(t1,data.loc[t1].close,marker='o',color='r',alpha=0.2)


# In[3419]:


len(events)


# In[ ]:





# In[3420]:


#events[('trgt','close')].plot()


# In[3421]:


events[('trgt','close')].plot()


# In[3422]:


d_bars.loc[t1].close.plot()


# In[ ]:





# In[ ]:





# In[3423]:


bins = getBins(events,d_bars['close'])


# In[3424]:


bins.bin.value_counts()


# In[3425]:


from sklearn.model_selection import cross_val_score


# In[3426]:


def adfuller_test(x):
    values=[]
    for i in x.columns:
        value = adfuller(x[i])
        p_value = value[1]
        if False:print(i,'>>>>',round(p_value,4),('\033[94m satisfied \033[0m') if p_value<0.05                       else '\033[91m not satisfied \033[0m')
        if p_value>0.05:values.append(i)
    return values


# In[ ]:





# In[3427]:


#v = adfuller_test(d_bars)


# In[ ]:





# In[3428]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


# In[3429]:


from imblearn.under_sampling import RandomUnderSampler


# In[3430]:


#under_sampling = RandomUnderSampler(sampling_strategy=0.7)


# In[3431]:


#x_over,y_over = under_sampling.fit_resample(train_x,train_y)


# In[3432]:


d_bars.drop(['Time','Num Ticks'],axis=1,inplace=True)


# In[3433]:


train_features = d_bars.loc[events.index]


# In[3434]:


bins.bin.value_counts()


# In[3435]:


train_features = train_features.drop(bins[bins.bin==0].index,axis=0)


# In[3436]:


bins = dropLabels(bins)


# In[3437]:


train_labels = bins['bin']


# In[3438]:


train_labels[train_labels==0]


# In[ ]:





# ## Sequential Bootstraping

# In[3439]:


numCoEvents=mpPandasObj(mpNumCoEvents,('molecule',events.index),1,                                         closeIdx=close.index,t1=events[('t1',0)])
numCoEvents=numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')] 
numCoEvents=numCoEvents.reindex(close.index).fillna(0) 


# In[3440]:


ev = mpPandasObj(mpSampleTW,('molecule',events.index),1,                     t1=events[('t1',0)],numCoEvents=numCoEvents)


# In[3441]:


mat = getIndMatrix(close.index,t1)


# In[3442]:


unique = getAvgUniqueness(mat)


# In[3443]:


#boot = seqBootstrap(mat,5)


# In[3444]:


plt.hist(unique,bins=50)
plt.show()


# In[3445]:


sam = mpPandasObj(mpSampleW,('molecule',events.index),1,  
            t1=events[('t1',0)],numCoEvents=numCoEvents,close=close)


# In[3446]:


unique.mean()


# ## Preprocessing

# In[3447]:


os.chdir('/Users/abhijitdeshpande/Documents/Project Files/Py files')


# In[3448]:


from selection import TechnicalIndicator
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score,classification_report


# In[3449]:


#train_features_ = fracDiff_FFD(train_features,0.5)


# In[3450]:


#train_features_.close.plot()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[3451]:


#adfuller_test(train_features_)


# In[3452]:


tech_data = TechnicalIndicator()(train_features)


# In[3453]:


labels = train_labels.loc[tech_data.loc[:train_labels.index[-1]].index]


# In[3454]:


train_features = tech_data.loc[:train_labels.index[-1]]


# In[ ]:





# In[3455]:


from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import auc,roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[3456]:


from sklearn.preprocessing import MinMaxScaler,StandardScaler
scale = StandardScaler().fit_transform(train_features)


# In[3457]:


from sklearn.decomposition import PCA
import seaborn as sns
pca = PCA()
pca_data = pca.fit_transform(scale)


# In[3458]:


#labels = labels.drop(train_features[train_features.close>15].index)


# In[3459]:


#train_features = train_features[train_features.close<15]


# In[3460]:


plt.figure(figsize=(10,6))
sns.scatterplot(pca_data[:,0],pca_data[:,1],hue=labels)
plt.xlabel('First Priciple Component')
plt.ylabel('Second Priciple Component')
plt.title('PCA of data')


# In[ ]:





# In[3461]:


from sklearn.inspection import permutation_importance


# In[3462]:


best_k = SelectFromModel(RandomForestClassifier(n_estimators=100,max_features=1))
best_k.fit(pca_data,labels)
cols = best_k.get_support()


# In[3463]:


cols.sum()


# In[3464]:


fea = train_features[train_features.columns[best_k.get_support()]]


# In[3465]:


rf = RandomForestClassifier(n_estimators=40)
rf.fit(train_features,labels)


# In[3466]:


results = permutation_importance(rf,train_features,labels,n_repeats=5)


# In[3467]:


sort_ = results.importances_mean.argsort()[::-1][:30]


# In[3468]:


fea2 = train_features[train_features.columns[sort_]]


# In[3469]:


len(fea2)


# In[ ]:





# In[ ]:





# In[3470]:


name


# In[3471]:


data_array = np.array(pca_data[:,:39])


# In[3472]:


labels = labels.apply(lambda x:0 if x<1 else 1)


# In[3473]:


size=0.7


# In[3474]:


import torch
import torch.nn as nn 


# In[3475]:


from preprocess import noise_mask, collate_unsuperv


# In[ ]:





# In[ ]:





# In[3476]:


labels[3175:].value_counts()


# In[3477]:


def data_prepare(data_array,labels,numDays,size):
    
    training_size = int(len(data_array)*size)
    val_size = training_size+int((len(data_array)-training_size)/2)
    idx_train, idx_val, idx_test = range(0,training_size), range(training_size,val_size),                                range(val_size,len(data_array)-numDays)
    
    x_train,length, days = [], len(data_array), numDays
    for i in range(0,(length-days),1):
        x_train.append(data_array[i:i+days])
    
    label = []
    for i in range(0,(length-days),1):
        label.append(labels[i+days])
        
    np.save('/Users/abhijitdeshpande/Desktop/npy_aapl.npy', x_train)
    x_train = torch.tensor(np.load('/Users/abhijitdeshpande/Desktop/npy_aapl.npy'))
    
    data_train = []
    for i in x_train[idx_train]:
        data_train.append(noise_mask(i,0.15))

    data_val = []
    for i in x_train[idx_val]:
        data_val.append(noise_mask(i,0.15))
    
    data_test = []
    for i in x_train[idx_test]:
        data_test.append(noise_mask(i,0.15))
        
    X, targets, masks, padding_masks = collate_unsuperv(x_train[idx_train],torch.tensor(data_train))
    
    X_val, targets_val, masks_val, padding_masks_val =                 collate_unsuperv(x_train[idx_val],torch.tensor(data_val))
    
    X_test, targets_test, masks_test, padding_masks_test =                 collate_unsuperv(x_train[idx_test],torch.tensor(data_test))
    
    train_data = []
    for i in idx_train:
        train_data.append([x_train[i],label[i],padding_masks[i]])
        
    val_data = []
    for i in idx_val:
        val_data.append([x_train[i],label[i],padding_masks_val[i-len(idx_train)]])
        
    test_data = []
    for i in idx_test:
        test_data.append([x_train[i],label[i],padding_masks_test[i-(len(idx_val)+len(idx_train))]])
        
    return train_data, val_data, test_data, x_train.shape[2], x_train.shape[1]


# In[3579]:


numDays = 5


# In[3580]:


train, val, test, feat_dim, max_len = data_prepare(data_array,labels,numDays,size)


# In[3581]:


from modules import TSTransformerEncoderLSTM, TSTransformerEncoderClass
from preprocess import train_model


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[3603]:


torch.manual_seed(10)
super_model = TSTransformerEncoderLSTM(feat_dim,max_len,4,0,3,
                                             8, 2,lstm_layers=1, hidden_layers=7,dropout=0.3)


# In[3604]:


supervised = train_model(super_model)
supervised.compile(loss=nn.CrossEntropyLoss(),optimizer=torch.optim.Adam,lr=5e-6)


# In[3607]:


supervised.train(train,val,batch_size=7,verbose=True,return_loss=True,epochs=3000,metric=True,early_stopping=20)


# In[ ]:





# In[3608]:


fp = supervised.analyze(test,color='Dark2')


# In[3586]:


np.argmax(fp,1)


# In[3316]:


np.array(labels)[3097:]


# In[ ]:





# In[ ]:





# In[ ]:





# In[3675]:


fpr,tpr,_=precision_recall_curve(labels[(len(train)+len(val))+days:],fp[:,1])


# In[3676]:


plt.plot(fpr,tpr)
# plt.plot([0,1],alpha=0.5)


# In[3677]:


auc(tpr,fpr)


# In[ ]:





# In[ ]:





# In[3320]:


def objective(trial):
    
    d_model=trial.suggest_int('d_model',1,10)
    #n_heads=trial.suggest_int('n_heads',0,2)
    #n_layers=trial.suggest_int('n_layers',1,5)
    dim_feedforward=trial.suggest_int('dim_feedforward',0,10)
    #lstm_layers=trial.suggest_int('lstm_layers',1,5)
    #hidden_layers=trial.suggest_int('hidden_layers',4,10)
    #lr=trial.suggest_loguniform('lr',1e-7,1e-3)
    #dropout=trial.suggest_discrete_uniform('dropout',0.1,0.7,0.05)
    #batch_size=trial.suggest_int('batch_size',5,11)
    
    torch.manual_seed(10)
    super_model = TSTransformerEncoderLSTM(feat_dim,max_len,d_model,0,1,
                                         dim_feedforward,2,2,7,0.3)
    supervised = train_model(super_model)
    supervised.compile(5e-5,loss = nn.CrossEntropyLoss(),optimizer=torch.optim.Adam)
    best_loss, auc = supervised.train(train, val, verbose=False, return_loss=False, epochs=3000, 
                                 batch_size=7,metric=True,early_stopping=100)
                          
    
    return best_loss, auc


# In[ ]:





# In[ ]:


import optuna


# In[260]:


study = optuna.multi_objective.create_study(["minimize", "maximize"])
study.optimize(objective, n_trials=10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[3609]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from tscv import GapKFold
from tscv import gap_train_test_split


# In[ ]:





# In[3610]:


x_train.shape


# In[3611]:


numDays = max_len


# In[3612]:


length = len(data_array)
days= numDays

x_train = []
for i in range(0,(length-days),1):
    x_train.append(data_array[i:i+days])
    
x_train = np.array(x_train)

x_train = x_train.reshape((len(data_array)-days),feat_dim*days)

label = []
for i in range(0,(length-days),1):
        label.append(labels[i+days])
        

X_train, X_test, y_train, y_test = gap_train_test_split(x_train, label, test_size=0.15, gap_size=0.001)


# In[3613]:


rf = RandomForestClassifier(n_estimators= 100,max_depth=3,max_features= 5,max_samples=unique.mean())


# In[3614]:


lr = LogisticRegression(solver= 'liblinear', C=15e-3)


# In[3615]:


clf = SVC(kernel='rbf',C=100,gamma='auto')


# In[3616]:


cv = GapKFold(n_splits=5, gap_before=10, gap_after=10)
scores = cross_val_score(rf, X_train, y_train, cv=cv,scoring='f1')


# In[3617]:


scores


# In[3624]:


model = svm


# In[3625]:


def results(model):
    model.fit(X_train,y_train)
    
    print(classification_report(y_test,model.predict(X_test)))
    print(confusion_matrix(y_test,model.predict(X_test)))


# In[3626]:


results(model)


# In[3623]:


svm = SVC(kernel='rbf',C=100,gamma='auto',probability=True)


# In[3632]:


precision, recall, thresholds = precision_recall_curve(y_test,svm.predict_proba(X_test)[:,1])


# In[3226]:


from sklearn.metrics import precision_recall_curve,recall_score


# In[3683]:


precision, recall, thresholds = roc_curve(y_test,svm.predict_proba(X_test)[:,1])


# In[3684]:


plt.plot(recall,precision)
#plt.plot([0,1],alpha=0.5)


# In[3685]:


accuracy_score(y_test,model.predict(X_test))


# In[3643]:


rf.predict_proba(X_test)


# In[3645]:


auc(recall,precision)


# In[2341]:


cross_val_score(clf, X_train, y_train, cv=cv,scoring='f1').mean()


# In[ ]:





# In[3164]:


def objective(trial):
    
    clf = trial.suggest_categorical('Classifier',['Logistic'])
    
    if clf=='RandomForest':
        n_estimators=trial.suggest_int('n_estimators',10,500)
        max_depth=trial.suggest_int('max_depth',1,20,log=True)
        max_features=trial.suggest_int('max_features',1,20)

        clf = RandomForestClassifier(n_estimators,max_depth=max_depth,
                                   max_features=max_features,max_samples=unique.mean())
    
    elif clf=='svm':
        C = trial.suggest_float('C',1e-5,1e5,log=True)   
        clf = SVC(kernel='rbf',C=C,gamma='auto')
    
    elif clf=='Logistic':
        #solver = trial.suggest_categorical('solver',['liblinear','saga','sag','lbfgs'])
        C = trial.suggest_float('C',1e-5,1e5,log=True)
        clf = LogisticRegression(C=C,solver='liblinear')
        
    return cross_val_score(clf, X_train, y_train, cv=cv,scoring='recall').mean()


# In[3165]:


import optuna


# In[3166]:


study = optuna.create_study(direction='maximize')


# In[3167]:


study.optimize(objective,n_trials=100)


# In[3168]:


study.best_params


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




