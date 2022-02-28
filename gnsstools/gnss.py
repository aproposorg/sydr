import configparser
from tqdm import tqdm
import pickle
from gnsstools.gnsssignal import GNSSSignal
from gnsstools.acquisition import Acquisition
from gnsstools.navigation import Navigation
from gnsstools.rffile import RFFile
from gnsstools.satellite import Satellite
from gnsstools.tracking import Tracking
from gnsstools.decoding import Decoding

class GNSS:
    def __init__(self, configfile, prnlist):
        # Initialise from config file
        config = configparser.ConfigParser()
        config.read(configfile)
    	
        self.configfile = configfile
        self.signalfile = config.get('DEFAULT', 'signalpath')
        
        self.signal_type = config.get('DEFAULT', 'signal_type')

        # Instanciate variables
        self.data_file = RFFile(self.configfile)
        self.signal = GNSSSignal(self.signalfile, self.signal_type)

        # Satellite
        self.prnlist = prnlist
        self.satelliteDict = {}
        for prn in prnlist:
            self.satelliteDict[prn] = Satellite()

        return
    
    # -------------------------------------------------------------------------

    def doAcquisition(self):
        for prn in tqdm(self.prnlist, desc="Acquisition progress"):
            acquisition = Acquisition(self.configfile, prn, self.signal)
            acquisition.acquire(self.data_file)
            self.satelliteDict[prn].setAcquisition(acquisition)
        return
    
    # -------------------------------------------------------------------------

    def doTracking(self, writeToFile=None, loadFromFile=None):
        if loadFromFile is not None:
            print("Tracking progress   : Loading from file... ", end='')
            with open(loadFromFile, 'rb') as f:
                self.satelliteDict = pickle.load(f)
            print("Done.")
        else:
            for prn in tqdm(self.prnlist, desc="Tracking progress   "):
                tracking = Tracking(self.configfile, self.satelliteDict[prn].getAcquisition())
                tracking.track(self.data_file)
                self.satelliteDict[prn].setTracking(tracking)
            if writeToFile is not None:
                with open(writeToFile, 'wb') as f:
                    pickle.dump(self.satelliteDict, f, pickle.HIGHEST_PROTOCOL)
        return
    
    # -------------------------------------------------------------------------

    def doDecoding(self):
        for prn in tqdm(self.prnlist, desc="Decoding progress   "):
            decoding = Decoding(self.satelliteDict[prn].getTracking())
            decoding.decode()
            self.satelliteDict[prn].setDecoding(decoding)
        return

    # -------------------------------------------------------------------------

    def doNavigation(self):
        
        # Compute pseudoranges
        self.navigation = Navigation(self.configfile)
        self.navigation.computePseudoranges(self.satelliteDict)

        return
    

