import sys, os
sys.path.append('/mnt/c/Users/vmangr/Documents/Code/sydr/') 

import numpy as np
import math
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
    
    trackingC.setSatellite(svid=2)
    trackingC.setInitialValues(estimatedFrequency=3700.0)
    trackingPython.setSatellite(svid=2)
    trackingPython.setInitialValues(estimatedFrequency=3700.0)

    trackingPython.run(rfdata)
    trackingC.run(rfdata)

    # Assert
    assert math.isclose(trackingC.correlatorResults[0], trackingPython.correlatorResults[0], rel_tol=1e-11) == True
    assert math.isclose(trackingC.correlatorResults[1], trackingPython.correlatorResults[1], rel_tol=1e-11) == True
    assert math.isclose(trackingC.correlatorResults[2], trackingPython.correlatorResults[2], rel_tol=1e-11) == True
    assert math.isclose(trackingC.correlatorResults[3], trackingPython.correlatorResults[3], rel_tol=1e-11) == True
    assert math.isclose(trackingC.correlatorResults[4], trackingPython.correlatorResults[4], rel_tol=1e-11) == True
    assert math.isclose(trackingC.correlatorResults[5], trackingPython.correlatorResults[5], rel_tol=1e-11) == True

    print("Unit test passed.")