
from gnsstools.gnss import GNSS
from gnsstools.analysis import Analysis

# Main program
configfile = './config/default_config.ini'
prnlist = [9]

gnss = GNSS(configfile)
analysis = Analysis()

# Acquisition
gnss.doAcquisition(prnlist)
analysis.acquisition(gnss.acquisition_results, corrMapsEnabled=False)

# Tracking
gnss.doTracking(prnlist, 36000)
analysis.tracking(gnss.tracking_results)

# Data decoding
gnss.doDecoding(prnlist)