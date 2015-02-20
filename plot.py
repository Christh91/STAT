from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt



# Load Data
input_file1= './hawaii_uas_nnrmse.csv'
pcrmse=np.genfromtxt(input_file1, delimiter=",")




# llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
# are the lat/lon values of the lower left and upper right corners
# of the map.
# resolution = 'c' means use crude resolution coastlines.




m = Basemap(projection='cea',llcrnrlat=12,urcrnrlat=30,\
            llcrnrlon=-166,urcrnrlon=-148,resolution='h')

cs=m.imshow(pcrmse)
cbar = plt.colorbar(cs, orientation='horizontal')
cbar.set_label('m/s')
m.drawstates()
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')

m.drawparallels(np.arange(12.,30,1.875),labels=[1,0,0,0],fontsize=10)
m.drawmeridians(np.arange(-166.,-148.,3.75),labels=[0,0,0,1],fontsize=10)
m.drawmapboundary(fill_color='aqua')
plt.title("NN's Calculated U10 RMSE Over WRF Outer Domain")
plt.show()