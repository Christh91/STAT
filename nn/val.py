__author__ = 'Daan Wierstra and Tom Schaul'


#Validate
import scipy as sci
import numpy as np
import cPickle as pickle
from math import sqrt
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation    import testOnSequenceData
from pybrain.structure import TanhLayer

#Declare inputs and traget 

input_file1 = './../data/orig/csv/uas_remap_Amon_CNRM-CM5_historical_r1i1p1_197901-200512.csv'
input_file2 = './../data/orig/csv/uas_remap_Amon_GFDL-CM3_historical_r1i1p1_197901-200512.csv'
input_file3 = './../data/orig/csv/uas_remap_Amon_IPSL-CM5A-LR_historical_r1i1p1_197901-200512.csv'
input_file4 = './../data/orig/csv/uas_remap_Amon_MIROC-ESM_historical_r1i1p1_197901-200512.csv'
input_file5 = './../data/orig/csv/uas_remap_Amon_MIROC5_historical_r5i1p1_197901-200512.csv'
input_file6 = './../data/orig/csv/uas_remap_Amon_NorESM1-M_historical_r1i1p1_197901-200512.csv'


target_file = './../data/orig/csv/uas_NCEP2_197901-200512.csv'

output_model_file = 'model_val.pkl'

#Load data

input1 = np.genfromtxt( input_file1, delimiter=",") 
input2 = np.genfromtxt( input_file2, delimiter=",")
input3 = np.genfromtxt( input_file3, delimiter=",")
input4 = np.genfromtxt( input_file4, delimiter=",")
input5 = np.genfromtxt( input_file5, delimiter=",")
input6 = np.genfromtxt( input_file6, delimiter=",")
target = np.genfromtxt( target_file, delimiter=",")

hidden_size = 100
epochs = 324
continue_epochs = 10	
validation_proportion = 0.185

j=0
x1=[[]]*324
x2=[[]]*324
x3=[[]]*324
x4=[[]]*324
x5=[[]]*324
x6=[[]]*324
y=[[]]*324
for i in range(0,324):
	t=i*94
	x1[i]=input1[t][j]
	x2[i]=input2[t][j]
	x3[i]=input3[t][j]
	x4[i]=input4[t][j]
	x5[i]=input5[t][j]
	x6[i]=input6[t][j]
	y[i]=target[t][j]

x = np.vstack((x1, x2, x3, x4, x5, x6))		
x=x.transpose()
y=np.asarray(y)
target=y.reshape(-1,1)

input_size = x.shape[1]
target_size = target.shape[1]

# prepare dataset

ds = SDS( input_size, target_size )
ds.setField( 'input', x )
ds.setField( 'target', target )

net = buildNetwork( input_size, hidden_size, target_size, hiddenclass=TanhLayer, bias= True, recurrent=True )
trainer = BackpropTrainer( net,ds,learningrate=0.0001 )

train_mse, validation_mse = trainer.trainUntilConvergence( verbose = True, validationProportion = validation_proportion, 
	maxEpochs = epochs, continueEpochs = continue_epochs )

pickle.dump( net, open( output_model_file, 'wb' ))





