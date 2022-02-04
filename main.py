import numpy as np
from gnsstools.gnss import GNSS
from gnsstools.analysis import Analysis
import gnsstools.ca as ca
from gnsstools.rffile import RFFile


# ## Main program
configfile = './config/default_config.ini'
prnlist = [21]

gnss = GNSS(configfile)
analysis = Analysis()

# # Acquisition
gnss.doAcquisition(prnlist)
analysis.acquisition(gnss.acquisition_results, "acquisition_L1CA", corr_maps_enabled=False)

## Tracking
gnss.doTracking(prnlist, 2000)