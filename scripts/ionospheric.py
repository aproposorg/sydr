import ionex
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import imageio
from tqdm import tqdm
import ftplib
from unlzw import unlzw

outputFolder = "/mnt/c/Users/vmangr/Documents/Datasets/ionosphere/"

startDOY = 1
stopDOY = 95
retrieveFile = False

# Download files
host = "gssc.esa.int"
if retrieveFile:
    with ftplib.FTP(host) as ftp:
        ftp.login()
        for doy in range(startDOY, stopDOY+1):
            ftp.cwd(f"/gnss/products/ionex/2022/{doy:03}/")
            filename = f"esrg{doy:03}0.22i.Z"
            with open(outputFolder+filename, 'wb') as f:
                ftp.retrbinary("RETR " + filename, f.write)

imageList = []
maxValues = []
time = []
for doy in range(startDOY, stopDOY+1): 
    # Uncompress
    filename_compressed = f"{outputFolder}esrg{doy:03}0.22i.Z"
    filename_uncompressed = f"{outputFolder}esrg{doy:03}0.22i"
    with open(filename_compressed, 'rb') as fcomp:
        compressed = fcomp.read()
        with open(filename_uncompressed, 'w+') as funcomp:
            funcomp.write(unlzw(compressed).decode('ascii'))
    
    # Create maps
    with open(filename_uncompressed) as file:
        inx = ionex.reader(file)
        for ionex_map in inx:

            #plt.figure(figsize=(15,6))
            #map = Basemap()
            #map.drawcoastlines()

            x = np.arange(\
                ionex_map.grid.longitude.lon1, \
                ionex_map.grid.longitude.lon2+1, \
                ionex_map.grid.longitude.dlon)
            
            y = np.arange(\
                ionex_map.grid.latitude.lat1, \
                ionex_map.grid.latitude.lat2-1, \
                ionex_map.grid.latitude.dlat)

            X, Y = np.meshgrid(x, y)
            Z = np.reshape(ionex_map.tec, X.shape)

            # fig = plt.pcolormesh(X,Y,Z, vmin=0, vmax=100)
            # plt.title(f"Ionosphere map on {ionex_map.epoch}")
            # cbar = plt.colorbar(fig)
            # plt.clim(0, 100) 
            # cbar.set_label('Total Electron Content [TECU]')
            # plt.show()
            # plt.savefig(f"{outputFolder}maps/{ionex_map.epoch}.png")
            # plt.close()

            print(f"{ionex_map.epoch}: {np.max(Z)}")

            time.append(ionex_map.epoch)
            maxValues.append(np.max(Z))
            #imageList.append(imageio.imread(f"{outputFolder}maps/{ionex_map.epoch}.png"))

# Make gif
#imageio.mimsave(f"{outputFolder}maps/mygif.gif", imageList, fps=8)
# 
plt.figure(figsize=(15,6))
plt.plot(time, maxValues)
plt.title("Maximum TEC per map")
plt.grid()
plt.xlabel("Time")
plt.ylabel("Total Electron Content [TECU]")
plt.ylim(30, 120)
plt.savefig(f"{outputFolder}maps/max_tec_evolution.png")
plt.close()

