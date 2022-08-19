# GPy_CatalystPred
The project contains the following data folders:

1. data : This folder contains all the excel files related to reactions (God file) , and descriptors of solvent, catalysts,esters, base and reaction conditions.

The designmats are generated using one hot encoding and combinations of chemical descriptors of ester and catalyst. Each designmat file
consists of descriptors and reaction conditions as the features, and yields as the target variable. Yield is the last column of designmats

Linear Regression can also be performed using the LinearKerenel in the GPy library. 

Codes folder contains:

1. 'GaussianProcess' code folder : Containing CatalystFunc.py and GPy.py . GPy.py imports data from designmats and functions from CatalystFuncs.
2. 'Graph kernel' folder : Jupyter Notebook yield_expt.ipynb for running Graphkernel based code on SMILES
3. 'NN' folder: Mathematica codes for Linear Regression and Neural Networks
