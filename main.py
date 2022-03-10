
from gnsstools.gnss import GNSS
from gnsstools.analysis import Analysis

# Main program
configfile = './config/default_config.ini'
#prnlist = range(1, 33)
prnlist = [2,3,4,6,9,11,29,31]
#prnlist = [3,6,9,15,18,21,22,26]

gnss = GNSS(configfile, prnlist)
analysis = Analysis()

loadPrevious = False

# Acquisition
gnss.doAcquisition(loadPrevious=loadPrevious)
analysis.acquisition(gnss.satelliteDict, corrMapsEnabled=False)

# Tracking
gnss.doTracking(loadPrevious=loadPrevious)
#analysis.tracking(gnss.satelliteDict)

# Data decoding
gnss.doDecoding(loadPrevious=loadPrevious)

# Navigation
gnss.doNavigation()
analysis.navigation(gnss.navigation)
analysis.measurements(gnss.satelliteDict)