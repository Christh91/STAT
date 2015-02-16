#Train
import scipy as sci
import numpy as np
import cPickle as pickle
from math import sqrt
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
#from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation    import testOnSequenceData
from pybrain.structure import TanhLayer




# Test input from netCDF4
#Declare inputs and traget 


input_file1 = '/Users/Chris/SLA/stat/data/orig/uas_remap_Amon_CNRM-CM5_historical_r1i1p1_197901-200512.nc'
input_file2 = '/Users/Chris/SLA/stat/data/orig/uas_remap_Amon_GFDL-CM3_historical_r1i1p1_197901-200512.nc'
input_file3 = '/Users/Chris/SLA/stat/data/orig/uas_remap_Amon_IPSL-CM5A-LR_historical_r1i1p1_197901-200512.nc'
input_file4 = '/Users/Chris/SLA/stat/data/orig/uas_remap_Amon_MIROC-ESM_historical_r1i1p1_197901-200512.nc'
input_file5 = '/Users/Chris/SLA/stat/data/orig/uas_remap_Amon_MIROC5_historical_r5i1p1_197901-200512.nc'
input_file6 = '/Users/Chris/SLA/stat/data/orig/uas_remap_Amon_NorESM1-M_historical_r1i1p1_197901-200512.nc'
target_file = '/Users/Chris/SLA/stat/data/orig/uas_NCEP2_197901-200512.nc'
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
		ex1 = input1.variables['uas'][:,j,i]
		ex2 = input2.variables['uas'][:,j,i]
		ex3 = input3.variables['uas'][:,j,i]
		ex4 = input1.variables['uas'][:,j,i]
		ex5 = input1.variables['uas'][:,j,i]
		ex6 = input1.variables['uas'][:,j,i]
		target = target1.variables['uwnd'][:,0,j,i]
		example = np.vstack((ex1, ex2, ex3, ex4, ex5, ex6))	
		example=example.transpose()
		target=target.reshape(-1,1)
		
#Compare MSE against MME MSE	(DOESNT WORK)!!!
#		
		mmemse=np.mean((target-(np.mean(example,1)))**2)
		print "MME MSE {}".format(mmemse)
		
# Get data shapes for pybrain	
		
		input_size = example.shape[1]
		target_size = target.shape[1]

# prepare dataset for pybrain

		ds = SDS( input_size, target_size )
		ds.setField( 'input', example )
		ds.setField( 'target', target )

# Declare some scalar variables
		hidden_size = 100
		epochs = 1000
		continue_epochs = 10	
		validation_proportion = 0.1825


# Initialize network

		net = buildNetwork( input_size, hidden_size, target_size, hiddenclass=TanhLayer, bias= True, recurrent=True)
		trainer = BackpropTrainer( net,ds,learningrate=0.001, weightdecay=0.1, batchlearning=False, verbose=False )



# For single epoch training and testing

#		print "training for {} epochs...".format( epochs )
#		for i in range( 1):
#			mse = trainer.train()
#			rmse = sqrt( mse )
#			print "training RMSE, epoch {}: {}".format( i + 1, rmse )
#			trainer.testOnData(verbose=True)
#	
		
# Outputs weight parameters but unsure how to use
		
		#	for mod in net.modules:
		#		print "Module:", mod.name
		#		if mod.paramdim > 0:
		#			print "--parameters:", mod.params
		#		for conn in net.connections[mod]:
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

# Train Network until convergence
		print "training unitl convergence on lon {} and lat {}".format(lon[i], lat[j])
		train_mse, validation_mse = trainer.trainUntilConvergence( verbose = True, validationProportion = validation_proportion, maxEpochs = epochs, continueEpochs = continue_epochs )
		print " train mse, validation mse".format(train_mse, validation_mse)

	
		
		pickle.dump( net, open( output_model, 'wb' ))