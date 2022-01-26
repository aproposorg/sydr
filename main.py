import numpy as np
from gnsstools.gnss import GNSS
import gnsstools.ca as ca


## Main program
configfile = './config/default_config.ini'
prnlist = [9, 10, 11, 12]

gnss = GNSS(configfile)

# Acquisition
gnss.doAcquisition(prnlist)