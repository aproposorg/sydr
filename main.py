
from gnsstools.gnss import GNSS
from gnsstools.analysis import Analysis

# Main program
configfile = './config/default_config.ini'
prnlist = [2,3,4,6,9,11]
#prnlist = [9, 15, 18, 21, 22]

gnss = GNSS(configfile, prnlist)
analysis = Analysis() 

# Acquisition
gnss.doAcquisition()
analysis.acquisition(gnss.satelliteDict, corrMapsEnabled=False)

# Tracking
#gnss.doTracking(writeToFile="./_results/satellites_results.pkl")
gnss.doTracking(loadFromFile="./_results/satellites_results.pkl")
#analysis.tracking(gnss.satelliteDict)

# Data decoding
gnss.doDecoding()

# Navigation
gnss.doNavigation()
analysis.navigation(gnss.navigation)