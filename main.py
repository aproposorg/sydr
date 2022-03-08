
from gnsstools.gnss import GNSS
from gnsstools.analysis import Analysis

# Main program
configfile = './config/default_config.ini'
#prnlist = range(1, 33)
prnlist = [2,3,4,6,9,11,31]
#prnlist = [3,6,9,15,18,21,22,26]

gnss = GNSS(configfile, prnlist)
analysis = Analysis()

# Acquisition
gnss.doAcquisition(loadPrevious=False)
analysis.acquisition(gnss.satelliteDict, corrMapsEnabled=False)

# Tracking
gnss.doTracking(loadPrevious=False)
analysis.tracking(gnss.satelliteDict)

# Data decoding
gnss.doDecoding(loadPrevious=False)

# Navigation
gnss.doNavigation()
analysis.navigation(gnss.navigation)