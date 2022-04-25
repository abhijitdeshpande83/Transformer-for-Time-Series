#!/usr/bin/env python
# coding: utf-8

# In[3]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings('ignore')


# In[4]:


import math
from typing import Optional, Any
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import accuracy_score
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer


# In[5]:


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else: return activation
    raise ValueError("activation should be relu/gelu, not {}".format(activation))

    
def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


# In[6]:


class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# In[7]:


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


# In[8]:


class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.feat_dim = feat_dim
        

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)


    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output) ## Z
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output



# In[9]:


class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.feat_dim = feat_dim
        

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)


    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output) ## Z
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output



# In[10]:


class TSTransformerEncoderClass(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoderClass, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Linear(d_model * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)

        return output


# In[11]:


class TSTransformerEncoderLSTM(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 hidden_layers =128,lstm_layers=1,
                 dropout=0.1, pos_encoding='learnable', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoderLSTM, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.hidden_layers = hidden_layers
        self.lstm_layers = lstm_layers

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.linear(num_classes)
        self.lstm = nn.LSTM(d_model,self.hidden_layers,self.lstm_layers,batch_first=True)

    def linear(self,num_classes):
        output_layer = nn.Linear(self.hidden_layers*self.lstm_layers,num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

        
    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        h0=c0= torch.zeros(self.lstm_layers,output.size(0),self.hidden_layers)
        output, (hn, cn) = self.lstm(output, (h0,c0))
        hn = hn.view(-1,self.hidden_layers*self.lstm_layers)
        output = self.output_layer(hn)
        
        return output


# In[12]:


class unsupervised_lstm_model(nn.Module):
    
    def __init__(self,model,d_model, max_len, num_classes):
        super(unsupervised_lstm_model, self).__init__()
        
        self.model = model
        self.linear = nn.Linear(d_model * max_len, num_classes)
        self.loss_module = nn.CrossEntropyLoss()
        self.output_layer = self.linear(num_classes)
        self.lstm = nn.LSTM(d_model,128,1,batch_first=True)
        
    def linear(self,num_classes):
        output_layer = nn.Linear(128,num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X, padding_masks):
        
        prediction = self.model(X, padding_masks)
        h0=c0= torch.zeros(1,prediction.size(0),128)
        output, (hn, cn) = self.lstm(prediction, (h0,c0)) 
        hn = hn.view(-1,128)
        output = self.output_layer(hn)
        return output       
         


# In[13]:


class unsupervised_model(nn.Module):
    
    def __init__(self,model,d_model, max_len, num_classes):
        super(unsupervised_model, self).__init__()
        
        self.model = model
        self.linear = nn.Linear(d_model * max_len, num_classes)
        self.loss_module = nn.CrossEntropyLoss()
        
    def forward(self, X, padding_masks):
        
        prediction = self.model(X, padding_masks)
        prediction = prediction.reshape(prediction.shape[0], -1)
        output = self.linear(prediction)
       
        return output


# In[14]:


class identity_block(nn.Module):
    def __init__(self,ni,nf,f=3):
        super(identity_block,self).__init__()
        
        self.conv1 = nn.Conv1d(ni, nf, kernel_size=1, stride=1,padding='valid')
        self.conv2 = nn.Conv1d(nf, nf, kernel_size=f,stride=1,padding='same')
        self.conv3 = nn.Conv1d(nf, nf, kernel_size=1,stride=1,padding='valid')
        
        self.batchnorm = nn.BatchNorm1d(nf)
        self.act = nn.ReLU()
        self.add = torch.add
        
    def forward(self,x):
    
        x_shortcut = x
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.act(x)
        
        x = self.conv3(x)
        x = self.batchnorm(x)
        
        x = self.add(x_shortcut,x)
        x = self.act(x)
        
        return x


# In[15]:


class convolution_block(nn.Module):
    def __init__(self, ni, nf, f, s=2):
        super(convolution_block,self).__init__()
        
        
        self.conv1 = nn.Conv1d(ni, nf, kernel_size=1, stride=s,padding='valid')
        self.conv2 = nn.Conv1d(nf, nf, kernel_size=f,stride=1,padding='same')
        self.conv3 = nn.Conv1d(nf, nf, kernel_size=1,stride=1,padding='valid')

        # expand channels for the sum if necessary
        self.shortcut = nn.Conv1d(ni, nf, 1, stride=s,padding='valid')
        self.batchnorm = nn.BatchNorm1d(nf)
        self.add = torch.add
        self.act = nn.ReLU()

    def forward(self, x):
        
        x_shortcut = x
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.act(x)
        
        x = self.conv3(x)
        x = self.batchnorm(x)
        x = self.act(x)
        
        x_shortcut = self.shortcut(x_shortcut)
        x_shortcut = self.batchnorm(x_shortcut)
        
        x = self.add(x, x_shortcut)
        x = self.act(x)
        
        return x


# In[16]:


class ResNet(nn.Module):
    def __init__(self, c_in, c_out):
        super(ResNet,self).__init__()
        nf = 64
        self.conv1 = convolution_block(c_in, nf, f=3,s=1)
        self.conv2 = convolution_block(nf, nf * 2, f=3)
        self.conv3 = convolution_block(nf * 2, nf * 4, f=3)
        self.conv4 = convolution_block(nf*4, nf*8, f=3)
        self.identity1 = identity_block(nf,nf)
        self.identity2 = identity_block(nf*2,nf*2)
        self.identity3 = identity_block(nf*4, nf*4)
        self.identity4 = identity_block(nf*8, nf*8)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = nn.Flatten()
        self.fc = nn.Linear(nf * 8, c_out)

    def forward(self, x, mask=None):
        x = self.conv1(x)
        x = self.identity1(x)
        x = self.conv2(x)
        x = self.identity2(x)
        x = self.conv3(x)
        x = self.identity3(x)
        x = self.conv4(x)
        x = self.identity4(x)
        x = self.squeeze(self.gap(x))
        return self.fc(x)

