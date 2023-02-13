# GPy_CatalystPred
Prediction of Ruthenium catalysed hydrogenation of esters is done using Gaussian Processes. GPy library has been used to implement Gaussian Processes. 

Datasets: 

1. 'data' folder contains all the excel files related to reactions curated from literature, descriptors of solvent, catalysts,esters, base and reaction conditions. 
2. designmats (design matrices) are compiled from DFT-calaculated descriptors. The designmats can be found in 'data/designmats' folder.
3. rdkit, ,morgan and maccs fingerprints have been calculated from rdkit library. The design matrices created using these fingerprints can be found in 'data/rdkit_mccas_morgan_fingerprints' folder

designmats are the imported data, which is then subdivided into input and output variables. Output variable is the 'yield', input variables are chemical properties of reactants, solvents, bases, reaction conditions etc.

Yield is the last column in all designmats.


An example for running the code:


We recommend using Anaconda for these codes. Create a new environment using the requirement.yml file:
conda env create --name myenv -f requirement.yml

Step 1: Go to codes/GaussianProcesses
Step 2: Download the GaussianProcesses folder
Step 3: Run GPy.py 

The errors and plots will be printed as follows:
<img width="945" alt="Screenshot 2023-02-13 at 19 01 04" src="https://user-images.githubusercontent.com/96231277/218550130-bea21fbf-b322-4591-ab2f-797f80e22f5e.png">

<img width="416" alt="Screenshot 2023-02-13 at 19 01 28" src="https://user-images.githubusercontent.com/96231277/218550246-997e6d7b-5425-4d1f-ab08-b79ae60b664c.png">


Linear Regression can also be performed using the LinearKerenel in the GPy library. 

Codes folder also contains:

1. 'GaussianProcess' code folder : Containing CatalystFunc.py and GPy.py . GPy.py imports data from designmats and various functions from CatalystFuncs.
2. 'Graph kernel' folder : Jupyter Notebook yield_expt.ipynb for running Graphkernel based code on SMILES
3. 'NN' folder: Mathematica codes for Neural Networks
