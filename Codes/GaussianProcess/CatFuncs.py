#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from itertools import product
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn import preprocessing
import random
import matplotlib.pyplot as plt
plt.interactive(True)

def load_data(path,s):
    data = pd.read_csv(path+'DMAT_'+str(s[0])+'_'+str(s[1])+'.csv',header = None)
    data = data.to_numpy()
    return data

def cutoff_data(data,cutoff):
    # yields are separated so that they can be used as target variable (Y) 
    yields=data[:,-1]
    
    # removing data points with yields <= cutoff. The resulting data is 'reduced_data' 
    pos=[x for x in range(0,len(yields)) if yields[x]>cutoff*100]
    reduced_data = data[pos]
    
    #calculating number of features (no. of columns in data - 1)
    numftrs=(data.shape)[1]-1
    
    #Separating data in X0 and Y0. X0 and Y0 consists of whole data set
    X0 = reduced_data[:,0:numftrs]
    Y0 = reduced_data[:,[numftrs]]
    
    # remove zero columns
    idx = np.argwhere(np.all(X0[..., :] == 0.0, axis=0))
    X0 = np.delete(X0, idx, axis=1)
    
    #number of data points (#rows) and updating number of features (#columns) after deleting zero columns
    
    numdat,numftrs=X0.shape  #numdata = number of data points; numftrs = number of features
    
    return numftrs,numdat,X0,Y0

def train_test_data(X,Y,train_frac):
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 1-train_frac,random_state=73)
    return X_train,X_test,Y_train,Y_test

def normalize_data(X0,Y0,X1,Y1,flag,numfeatures):
    if flag==0:
        mean_input, stdev_input, mean_output, stdev_output = np.zeros(numfeatures), np.ones(numfeatures), np.array(0.0), np.array(1.0);
        return X0, Y0, mean_input, stdev_input, mean_output, stdev_output, X1, Y1
    
    if flag==2:
        mean_input, stdev_input, mean_output, stdev_output = np.zeros(numfeatures), np.ones(numfeatures), np.array(0.0), np.array(1.0);
        mean_output=np.mean(Y0);
        
        Y0=(Y0-mean_output)
        Y1=(Y1-mean_output)
        return X0, Y0, mean_input, stdev_input, mean_output, stdev_output, X1, Y1
    
    if flag==3:
        mean_input, stdev_input, mean_output, stdev_output = np.zeros(numfeatures), np.ones(numfeatures), np.array(0.0), np.array(1.0);
        min_max_scaler = preprocessing.MinMaxScaler()
        X0 = min_max_scaler.fit_transform(X0)
        X1 = min_max_scaler.fit_transform(X1)
        Y0 = Y0/100
        Y1 = Y1/100
        return X0, Y0, mean_input, stdev_input, mean_output, stdev_output, X1, Y1
        
    scalerY = preprocessing.StandardScaler()
    Y0 = scalerY.fit_transform(Y0)
    Y1 = scalerY.transform(Y1)
    mean_input = np.zeros(numfeatures)
    mean_output = scalerY.mean_
    stdev_input = np.ones(numfeatures)
    stdev_output = np.sqrt(scalerY.var_)
    
    return X0, Y0, mean_input, stdev_input, mean_output, stdev_output, X1, Y1

def original_scale(mu,sigma,vec):
    return sigma*vec+mu

def mean_percent_error(v,v_true,mu,sigma):
    #v is the prediction , v_true is the true data
    #mu and sigma is calculated from 'func: normalize_train_IO_data' using training data
    
    v, v_true = original_scale(mu,sigma,v), original_scale(mu,sigma,v_true) 
    err=100*np.mean(abs((v-v_true)/v_true))
    return err

def rmse_mae_r2(v,v_true,mu,sigma):
    v, v_true = original_scale(mu,sigma,v), original_scale(mu,sigma,v_true)
    return mean_squared_error(v_true,v, squared = False),mean_absolute_error(v_true,v),r2_score(v_true,v)

def plot_save_fig (filename,y1,y2,y3,s,cutoff):
    l = np.max(np.array([np.max(y1),np.max(y2)]))
    m = np.min(np.array([np.min(y1),np.min(y2)]))
    if m>0:
        m=0
    xplot = np.array([m,l])
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'Serif'
    plt.xlim(0,100)
    plt.ylim(m,l)
    plt.xlabel("y_real")
    plt.ylabel("y_predict")
    plt.plot(xplot,xplot, linestyle = 'dotted', color = 'r')
    plt.plot(xplot,xplot*0.9, color = 'b')
    plt.plot(xplot,xplot*1.1, color = 'b')
    plt.errorbar(y1, y2.reshape(y1.shape),yerr = y3.reshape(y1.shape),fmt ='.',color = 'r', ecolor= 'b', capsize = 3.0, elinewidth = 0.5)
    plt.legend([r'$\bf{y=x}$',r'$\bf{y=0.9x}$',r'$\bf{y=1.1x}$',r'$\bf{Prediction}$'],fontsize = 10,loc ='upper left')
    plt.title(s)
    #Uncomment the line below to save the plot
    plt.savefig(filename,bbox_inches = 'tight',dpi=600,transparent = 'True',facecolor = 'white')
    plt.show()

def model_train_test(model,X_train, X_test,maxfeval,optimize):
    if optimize == True:
        model.optimize(messages = True, max_f_eval = maxfeval)
    return model.predict(X_train),model.predict(X_test)

def data_prep(path,s,cutoff,train_frac,normalization_flag):
    dataset =  load_data(path,s)
    numftrs, numdata, X, Y = cutoff_data(dataset,cutoff)
    X_train,X_test,Y_train, Y_test = train_test_data(X,Y,train_frac = train_frac)
    X_train, Y_train, mean_X, stdev_X, mean_Y, stdev_Y, X_test, Y_test = normalize_data(X_train, Y_train,X_test, Y_test,normalization_flag,numftrs)
    return numftrs,X_train, Y_train, mean_X, stdev_X, mean_Y, stdev_Y, X_test, Y_test


# In[ ]:




