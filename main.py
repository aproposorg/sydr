import numpy as np
from gnsstools.gnss import GNSS
from gnsstools.analysis import Analysis
import gnsstools.ca as ca
from gnsstools.rffile import RFFile


# ## Main program
configfile = './config/default_config.ini'
prnlist = [2,3,4,6,9,11,29,31]

gnss = GNSS(configfile)
analysis = Analysis()

# # Acquisition
gnss.doAcquisition(prnlist)
analysis.acquisition(gnss.acquisition_results, corrMapsEnabled=False)

## Tracking
gnss.doTracking(prnlist, 60000)
analysis.tracking(gnss.tracking_results)