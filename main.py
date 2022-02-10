import numpy as np
from gnsstools.gnss import GNSS
from gnsstools.analysis import Analysis
import gnsstools.ca as ca
from gnsstools.rffile import RFFile


# Main program
configfile = './config/default_config.ini'
prnlist = [9]

gnss = GNSS(configfile)
analysis = Analysis()

# Acquisition
gnss.doAcquisition(prnlist)
analysis.acquisition(gnss.acquisition_results, corrMapsEnabled=True)

# Tracking
gnss.doTracking(prnlist, 36000)
analysis.tracking(gnss.tracking_results)

# Data decoding
gnss.doDecoding(prnlist)