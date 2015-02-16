#Train
import scipy as sci
import numpy as np
import cPickle as pickle
from math import sqrt
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation    import testOnSequenceData
from pybrain.structure import TanhLayer




# Test input from netCDF4
#Declare inputs and traget 


input_file1 = '/Users/Chris/SLA/stat/data/orig/vas_remap_Amon_CNRM-CM5_historical_r1i1p1_197901-200512.nc'
input_file2 = '/Users/Chris/SLA/stat/data/orig/vas_remap_Amon_GFDL-CM3_historical_r1i1p1_197901-200512.nc'
input_file3 = '/Users/Chris/SLA/stat/data/orig/vas_remap_Amon_IPSL-CM5A-LR_historical_r1i1p1_197901-200512.nc'
input_file4 = '/Users/Chris/SLA/stat/data/orig/vas_remap_Amon_MIROC-ESM_historical_r1i1p1_197901-200512.nc'
input_file5 = '/Users/Chris/SLA/stat/data/orig/vas_remap_Amon_MIROC5_historical_r5i1p1_197901-200512.nc'
input_file6 = '/Users/Chris/SLA/stat/data/orig/vas_remap_Amon_NorESM1-M_historical_r1i1p1_197901-200512.nc'
target_file = '/Users/Chris/SLA/stat/data/orig/vas_NCEP2_197901-200512.nc'
output_model = 'model.pkl'

#Load data examples

input1 = Dataset(input_file1, 'r')
input2 = Dataset(input_file2, 'r')
input3 = Dataset(input_file3, 'r')
input4 = Dataset(input_file4, 'r')
input5 = Dataset(input_file5, 'r')
input6 = Dataset(input_file6, 'r')
target1 = Dataset(target_file, 'r')

#Load lat and lon as arrays for indexing

lon = input1.variables['lon'][:]
lat = input1.variables['lat'][:]



#Get desired variable into numpy array
for i in range(103,114):
	for j in range(30,41):		
		ex1 = input1.variables['vas'][:,j,i]
		ex2 = input2.variables['vas'][:,j,i]
		ex3 = input3.variables['vas'][:,j,i]
		ex4 = input1.variables['vas'][:,j,i]
		ex5 = input1.variables['vas'][:,j,i]
		ex6 = input1.variables['vas'][:,j,i]
		target = target1.variables['vwnd'][:,0,j,i]
		example = np.vstack((ex1, ex2, ex3, ex4, ex5, ex6))	
		example=example.transpose()
		target=target.reshape(-1,1)
		
#Compare MSE against MME MSE	(DOESNT WORK)!!!
#		
		mmemse=np.mean((target-(np.mean(example,1)))**2)
		print "MME MSE {}".format(mmemse)
		
# Get data shapes for pybrain	

input_size = x.shape[1]
target_size = target.shape[1]

# prepare dataset

ds = SDS( input_size, target_size )
ds.setField( 'input', x )
ds.setField( 'target', target )

# init and train

net = buildNetwork( input_size, hidden_size, target_size, bias= True, recurrent=True)
trainer = BackpropTrainer( net,ds,learningrate=0.00001, batchlearning=False, verbose=False )

print "training for {} epochs...".format( epochs )

#for i in range( 3):
#	mse = trainer.train()
#	rmse = sqrt( mse )
#	print "training RMSE, epoch {}: {}".format( i + 1, rmse )
#	trainer.testOnData(verbose=True)
	
#	for mod in net.modules:
#		print "Module:", mod.name
#		if mod.paramdim > 0:
#			print "--parameters:", mod.params
#		for conn in net.connections[mod]:
#			print "-connection to", conn.outmod.name
#			if conn.paramdim > 0:
#				print "- parameters", conn.params
#		if hasattr(net, "recurrentConns"):
#			print "Recurrent connections"
#			for conn in net.recurrentConns:             
#				print "-", conn.inmod.name, " to", conn.outmod.name
#				if conn.paramdim > 0:
#					print "- parameters", conn.params
#					for cc in range(len(conn.params)):
#						print conn.whichNeuron[cc], conn.params[cc]
#						
#	#pp = pprint.PrettyPrinter(indent=4) 
#	#pp.pprint( net['hidden0'].__dict__)	



train_mse, validation_mse = trainer.trainUntilConvergence( verbose = True, validationProportion = validation_proportion, 
	maxEpochs = epochs, continueEpochs = continue_epochs )
	

	
		
pickle.dump( net, open( output_model, 'wb' ))