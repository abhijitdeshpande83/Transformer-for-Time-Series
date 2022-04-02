#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings('ignore')


# In[3]:


import math
from typing import Optional, Any
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import accuracy_score
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, auc, precision_recall_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,roc_curve, roc_auc_score

# In[2]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          color='viridis'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=color)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[4]:


def noise_mask(X, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)
    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])

    return mask


def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked
    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def compensate_masking(X, mask):
    """
    Compensate feature vectors after masking values, in a way that the matrix product W @ X would not be affected on average.
    If p is the proportion of unmasked (active) elements, X' = X / p = X * feat_dim/num_active
    Args:
        X: (batch_size, seq_length, feat_dim) torch tensor
        mask: (batch_size, seq_length, feat_dim) torch tensor: 0s means mask and predict, 1s: unaffected (active) input
    Returns:
        (batch_size, seq_length, feat_dim) compensated features
    """

    # number of unmasked elements of feature vector for each time step
    num_active = torch.sum(mask, dim=-1).unsqueeze(-1)  # (batch_size, seq_length, 1)
    # to avoid division by 0, set the minimum to 1
    num_active = torch.max(num_active, torch.ones(num_active.shape, dtype=torch.int16))  # (batch_size, seq_length, 1)
    return X.shape[-1] * X / num_active


# In[11]:


def collate_unsuperv(data, mask, max_len=None, mask_compensation=False):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, mask).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - mask: boolean torch tensor of shape (seq_length, feat_dim); variable seq_length.
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 ignore (padding)
    """

    batch_size = len(data)
    features, masks = data, mask

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    target_masks = torch.zeros_like(X,
                                    dtype=torch.bool)  # (batch_size, padded_length, feat_dim) masks related to objective
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        target_masks[i, :end, :] = masks[i][:end, :]

    targets = X.clone()
    X = X * target_masks  # mask input
    if mask_compensation:
        X = compensate_masking(X, target_masks)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep
    target_masks = ~target_masks  # inverse logic: 0 now means ignore, 1 means predict
    return X, targets, target_masks, padding_masks


# In[12]:


class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.
        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered
        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)


# In[7]:


class train_model():
    
    def __init__(self,model):
        self.model = model
        self.validation_loss = []
        self.training_loss = []
        
    def compile(self, lr=1e-3, loss=nn.CrossEntropyLoss(),optimizer=torch.optim.Adam):  
        self.loss_module = loss
        self.optimizer = optimizer(self.model.parameters(),lr=lr)

   
    def fit(self, data, batch_size):

        self.model = self.model.train()
        total_loss = 0
        total_batch_accuracy = 0

        for i, batch in enumerate(DataLoader(data, batch_size)):
            x_train, y_train, masks = batch
            prediction = self.model(x_train.float(),masks)
            loss = self.loss_module(prediction, y_train)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            total_loss += loss
            mean_batch_loss = total_loss/(i+1)

            predictions = torch.argmax(prediction, 1)
            batch_accuracy = accuracy_score(y_train, predictions)

            
            total_batch_accuracy += batch_accuracy 
            mean_bach_accuracy = total_batch_accuracy/(i+1)
            
        return mean_batch_loss.detach().numpy(), mean_bach_accuracy
        
    def evaluate(self, data, batch_size):

        total_batch_accuracy = 0
        total_loss = 0
        batch_auc = []
        self.model = self.model.eval()
            
        for i, batch in enumerate(DataLoader(data, batch_size)):
            x_test, y_test, masks = batch
            prediction = self.model(x_test.float(),masks)
            probs = nn.Softmax()(prediction).detach().numpy()
            loss = self.loss_module(prediction, y_test)

            total_loss += loss
            mean_batch_loss = total_loss/(i+1)

            predictions = torch.argmax(prediction, 1)
            batch_accuracy = accuracy_score(y_test, predictions)
            
            precision, recall,_ = precision_recall_curve(y_test,probs[:,1])
            auc_ = auc(recall,precision)
            batch_auc.append(auc_)

            total_batch_accuracy += batch_accuracy 
            mean_bach_accuracy = total_batch_accuracy/(i+1)
        total_auc = np.nansum(batch_auc)/len(batch_auc)

        return mean_batch_loss.detach().numpy(), total_auc, mean_bach_accuracy
         
    def train(self,train_data,val_data,epochs=100,batch_size=7,verbose=True,return_loss=True,metric=False,early_stopping=10):
               
        best_loss = np.inf
        early_stop_counter = 0
        self.batch_size = 2**batch_size
        
        for epoch in range(epochs):
    
            loss, batch_accuracy = self.fit(train_data, self.batch_size)
            val_loss, auc_, val_batch_accuracy = self.evaluate(val_data, batch_size)
            
            self.training_loss.append(loss)
            self.validation_loss.append(val_loss)
           
            if val_loss < best_loss:
                best_loss = val_loss
                
            else:early_stop_counter+=1
            
            if early_stop_counter>early_stopping:
                break
            
            if verbose:
                print('\033[94m'+'Epoch: {}'.format(epoch)+'\033[0m'+' {}%'.format(round(batch_accuracy*100,3)),
                  '\033[91m'+'loss:'+'\033[0m'+' {}'.format(round(float(loss),4)),
                  '\033[94m Val:'+'\033[0m'+' {}'.format(round(val_batch_accuracy*100,3)),
                  '\033[91m'+'val_loss:'+'\033[0m'+' {}'.format(round(float(val_loss),4))
                 )
      

        if return_loss:
            plt.plot(self.training_loss)
            plt.plot(self.validation_loss)
            plt.legend()
        
        if metric:
            return best_loss.item(), auc_
        
    def analyze(self,data,color='viridis'):
        self.data = data
        self.model = self.model.eval()
        
        for i, batch in enumerate(DataLoader(self.data, 100000)):
            x_test, y_test, masks = batch
            prediction = self.model(x_test.float(), masks)
            probs = nn.Softmax()(prediction).detach().numpy()
            
            predictions = torch.argmax(prediction,1)
            accuracy = accuracy_score(predictions, y_test)
            print('\033[92m'+'\nAccuracy: '+'\033[0m',accuracy)
            print('\n'+'\033[0m'+         'Classification Report:\n')
            print(classification_report(y_test,predictions))
            print('\n'+'\033[0m'+         'Confusion Matrix:\n')
            cm = confusion_matrix(y_test,predictions)
            print('\033[91m'+' '*12+'Total Data Samples:'+'\033[0m', len(y_test),'\n')
            print('\033[94m'+' '*12+'Class 0:'+'\033[0m',(torch.unique(y_test, return_counts=True)[1][0]).item())
            print('\033[94m'+' '*12+'Class 1:'+'\033[0m',(torch.unique(y_test, return_counts=True)[1][1]).item(),'\n'+'\033[0m')

            plot_confusion_matrix(cm,[0,1],color=color)

            print('\n')

            try:
                precision, recall,_ = precision_recall_curve(y_test,probs[:,1])
                #roc_auc = roc_auc_score(y_test,prediction[:,1])
                auc_ = auc(recall,precision)
                plt.figure()
                plt.plot(recall, precision, label='(Area = {:.3f})'.format(auc_))
                #plt.plot([0,1],marker='.')
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.title('ROC curve')
                plt.legend(loc='best')
                plt.show()

            except Exception as e:print(e)
            return probs
        
# In[8]:


class pretrain():
    
    def __init__(self,model):
        self.model = model
        self.loss_module = nn.MSELoss()
        self.epoch_metrics = {}
        self.total_loss = 0
        self.batch_loss = []
                
        
    def train(self,data, batch_size=20,epochs=100,verbose=True,return_loss=False):
        self.model = self.model.train()
        self.data = data
        self.verbose = verbose
        self.return_loss = return_loss
        self.batch_size = batch_size
        self.epochs = epochs
  

        for epoch in range(self.epochs):
    
            total_batch_loss = 0
            total_loss = 0
    
            for i, batch in enumerate(DataLoader(self.data,self.batch_size)):
                masked_data, targets, target_masks, padding_masks = batch
                prediction = self.model(masked_data,padding_masks)
                masked_prediction = torch.masked_select(prediction,target_masks)
                masked_true = torch.masked_select(targets, target_masks)
                optimizer = optim.Adam(self.model.parameters())
                loss = self.loss_module(masked_prediction, masked_true)
        
                total_loss +=loss
                mean_batch_loss = total_loss/(i+1)
        
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                optimizer.step()
        
                metrics = {'mean batch loss':mean_batch_loss.item()}
        
        
            self.batch_loss.append(mean_batch_loss.detach().numpy())
            if self.verbose:
                print('\033[94m'+'Epoch {}:'.format(epoch)+'\033[0m'+' {}'.format(round(mean_batch_loss.item(),3)))
            self.total_loss += mean_batch_loss
            self.epoch_metrics['Epoch'] = epoch
            self.epoch_metrics['Loss'] = self.total_loss/(epoch+1)
        
        if self.return_loss:
            return plt.plot(self.batch_loss)
        
        
    def evaluate(self,data,batch_size=20):
            self.data = data
            self.batch_size = batch_size
            total_batch_loss = 0
            self.model = self.model.eval()
            
            for i, batch in enumerate(DataLoader(self.data, self.batch_size)):
                masked_data, targets, target_masks, padding_masks = batch
                prediction = self.model(masked_data, padding_masks)
                masked_prediction = torch.masked_select(prediction,target_masks)
                masked_true = torch.masked_select(targets, target_masks)
                
                batch_loss = self.loss_module(masked_prediction, masked_true)
                total_batch_loss += batch_loss
                mean_bach_loss = total_batch_loss/(i+1)
            metrics = {'mean batch loss':mean_bach_loss.item()}
                
            return metrics


# In[10]:


class data_preprocess():
    
    def get_data(self,name,path='/Users/abhijitdeshpande/Documents/Project Files/Data Files/', days=30,test_size=0.3):
    
        #import data
        loaded_data = pd.read_csv(path+name+'.csv')
        features = np.array(loaded_data.drop('out',axis=1))
        label = np.array((loaded_data[['out']]))
        
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        scaler = StandardScaler()
        features = scaler.fit_transform(features)


    
        length = len(loaded_data)
        days = days
        data = []
        for i in range(0,(length-days)):
            data.append(features[i:i+days])
    
        np.save('/Users/abhijitdeshpande/Documents/Project Files/npy_data/npy_data_'+name+'.npy',data)
    
        self.features =  torch.tensor(np.load('/Users/abhijitdeshpande/Documents/Project Files/'                                                                                   'npy_data/npy_data_'+name+'.npy',allow_pickle=True))
    
        lab = []
        for i in range(0,(length-days)):
            lab.append(label[i+days])
        
        self.label = torch.tensor(lab).squeeze()
        self.test_size = test_size
        
    def split(self):
        
        feature_len = len(self.features)
        train_size = int(abs(feature_len*(1-self.test_size)))
        val_size = int(abs(feature_len*(self.test_size/2.)))
    
    
        self.idx_train = range(train_size)
        self.idx_val = range(train_size, train_size+val_size)
        self.idx_test = range(train_size+val_size, feature_len)
        
        print('\033[1m'+'Total Samples:',feature_len)
        print('\033[92m'+'Training Samples:',len(self.idx_train))  
        print('\033[92m'+'Validation Samples:',len(self.idx_val))  
        print('\033[92m'+'Testing Samples:',len(self.idx_test)) 
      
    def creat_masks(self):
        
        data_train = []
        for i in self.features[self.idx_train]:
            data_train.append(noise_mask(i, 0.15))
    
        data_val = []
        for i in self.features[self.idx_val]:
            data_val.append(noise_mask(i, 0.15))
    
        data_test = []
        for i in self.features[self.idx_test]:
            data_test.append(noise_mask(i, 0.15))
            
            
        self.X, self.targets, self.target_masks, self.padding_masks =                                   collate_unsuperv(self.features[self.idx_train], torch.tensor(data_train))
        
        self.X_val, self.targets_val, self.target_masks_val, self.padding_masks_val =                         collate_unsuperv(self.features[self.idx_val], torch.tensor(data_val))
    
        self.X_test, self.targets_test, self.target_masks_test, self.padding_masks_test =                         collate_unsuperv(self.features[self.idx_test], torch.tensor(data_test))
        
         
    def prepare_data(self,supervised=True, unsupervised=False):
        
        self.split()
        self.creat_masks()
            
        x_train = self.features[self.idx_train]
        x_val = self.features[self.idx_val]
        x_test = self.features[self.idx_test]
        feat_dim = self.features.shape[2]
        max_len = self.features.shape[1]
        
        if supervised:
    
            train_data = []
            for i in range(len(self.idx_train)):
                train_data.append([x_train[i],self.label[self.idx_train][i],self.padding_masks[i]])
        
            val_data = []
            for i in range(len(self.idx_val)):
                val_data.append([x_val[i],self.label[self.idx_val][i],self.padding_masks_val[i]])
        
            test_data = []
            for i in range(len(self.idx_test)):
                test_data.append([x_test[i],self.label[self.idx_test][i],self.padding_masks_test[i]])
        
            
            return train_data, val_data, test_data, feat_dim, max_len
        
        if unsupervised:

            train_mask_data = []
            for i in range(len(self.idx_train)):
                train_mask_data.append([self.X[i],self.targets[i],self.target_masks[i],self.padding_masks[i]])
        
            val_mask_data = []
            for i in range(len(self.idx_val)):
                val_mask_data.append([self.X_val[i],self.targets_val[i],self.target_masks_val[i],self.padding_masks_val[i]])
        
            test_mask_data = []
            for i in range(len(self.idx_test)):
                test_mask_data.append([self.X_test[i],self.targets_test[i],self.target_masks_test[i],self.padding_masks_test[i]])
            
            
        
            return train_mask_data, val_mask_data, test_mask_data, feat_dim, max_len
            

