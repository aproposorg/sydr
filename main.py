import numpy as np
from gnsstools.gnss import GNSS
from gnsstools.analysis import Analysis
import gnsstools.ca as ca


## Main program
configfile = './config/default_config.ini'
prnlist = range(1, 33)

gnss = GNSS(configfile)
analysis = Analysis()

# Acquisition
gnss.doAcquisition(prnlist)
analysis.acquisition(gnss.acquisition_results, "acquisition_L1CA")