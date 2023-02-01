import sys, os
sys.path.append('/mnt/c/Users/vmangr/Documents/Code/sydr/') 

import numpy as np
from core.signal.gnsssignal import GNSSSignal, GNSSSignalType
from core.signal.rfsignal import RFSignal
from core.tracking.tracking_epl_c import Tracking as TrackingInC
from core.tracking.tracking_epl import Tracking as TrackingInPython

if __name__ == "__main__":

    rfConfigFile = '/mnt/c/Users/vmangr/Documents/Code/sydr/config/rf.ini'
    gnssSignalConfigFile = '/mnt/c/Users/vmangr/Documents/Code/sydr/config/signals/GPS_L1_CA.ini'
    
    rfSignal = RFSignal(rfConfigFile)
    gnssSignal = GNSSSignal(gnssSignalConfigFile, GNSSSignalType.GPS_L1_CA) 
    
    trackingC = TrackingInC(rfSignal, gnssSignal)
    trackingPython = TrackingInPython(rfSignal, gnssSignal)

    # load test data
    with open('/mnt/c/Users/vmangr/Documents/Code/sydr/core/unitTest/data/i_rfdata.txt') as f:
        rfdata = np.loadtxt(f, dtype=np.complex_)

    with open('/mnt/c/Users/vmangr/Documents/Code/sydr/core/unitTest/data/iSignal.txt') as f:
        iSignal = np.loadtxt(f)

    with open('/mnt/c/Users/vmangr/Documents/Code/sydr/core/unitTest/data/qSignal.txt') as f:
        qSignal = np.loadtxt(f)
    
    print(rfdata)
    print(iSignal)
    print(qSignal)
    
    trackingC.setSatellite(svid=2)
    trackingC.run(rfdata)
    

    print('whatever')