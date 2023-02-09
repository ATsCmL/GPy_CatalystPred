#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import GPy
from CatFuncs import *
from sklearn.decomposition import PCA
from sklearn import preprocessing
from itertools import product



path = "data/designmats/"
designmatindices = [(8,8)] 

#parameters for importing data:
yield_cutoffs = [0.0]
train_frac = 0.7

#normalizarion
normalization_flag = 5 #0 for no normalisation; 1 for standardised data; 2 for mean output on training set = 0


for i in range(len(designmatindices)):
    for yield_cutoff in yield_cutoffs:
        s = designmatindices[i]
        
        #data preparation:
            
        numftrs,X_train,Y_train,mu_X,stdev_X,mu_Y,stdev_Y,X_test,Y_test = data_prep(path,s,yield_cutoff,train_frac,normalization_flag)
            
        
        ##############################
        #GPy model training and testing: 
        ###############################

        #kernel

            
        kernel =  GPy.kern.Matern52(numftrs,ARD=True)+GPy.kern.Bias(numftrs)
            
        #simple GP model with kernel = kernel
        m = GPy.models.GPRegression(X_train,Y_train,kernel = kernel)
         
            
        #maximum iterations    
        maxfeval = 100000
        
        #prediction
        [pred_mean_train, var_train], [pred_mean_test, var_test] = model_train_test(m,X_train,X_test,maxfeval,optimize = True)
         
        #error calculations
        std_train, std_test = np.sqrt(var_train), np.sqrt(var_test)
        err_train = mean_percent_error(pred_mean_train,Y_train,mu_Y,stdev_Y)
        err_test = mean_percent_error(pred_mean_test,Y_test,mu_Y,stdev_Y)
        mse_train, mae_train, r2_train = rmse_mae_r2(pred_mean_train,Y_train,mu_Y,stdev_Y)
        mse_test, mae_test, r2_test = rmse_mae_r2(pred_mean_test,Y_test,mu_Y,stdev_Y)
        

        
        print('############################## \nResults:\n##############################')
        print ('Yield cutoff:',yield_cutoff)
        print('Train_frac:',train_frac)
        print('Designmat:',s)
        print('Train error:',err_train)
        print('Test error:',err_test)
        print('Train mae error:',mae_train)
        print('Test mae error:',mae_test)
        print('Train mse error:',mse_train)
        print('Test mse error:',mse_test)
        print('Train r2 error:',r2_train)
        print('Test r2 error:',r2_test)
        m.kern.plot_ARD()
        
        #plot and save figures
        plot_save_fig('filename for training', original_scale(mu_Y,stdev_Y,pred_mean_train.flatten()),original_scale(mu_Y,stdev_Y,Y_train.flatten()),std_train.flatten(),s,0.5)
        plot_save_fig('filename for test', original_scale(mu_Y,stdev_Y,pred_mean_test.flatten()),original_scale(mu_Y,stdev_Y,Y_test.flatten()),std_test.flatten(),s,0.5)
        
        

