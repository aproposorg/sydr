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
        self.dumpfile   = config.get('DEFAULT', 'dump_file')
        self.signalfile = config.get('DEFAULT', 'signalpath')
        self.signalType = config.get('DEFAULT', 'signal_type')

        self.data_file = RFFile(self.configfile)
        self.signal = GNSSSignal(self.signalfile, self.signalType)

        # Satellite
        self.prnlist = prnlist
        self.satelliteDict = {}
        for prn in prnlist:
            self.satelliteDict[prn] = Satellite()

        return
    
    # -------------------------------------------------------------------------

    def doAcquisition(self, loadPrevious=False):
        if loadPrevious:
            print("Acquisition progress: Loading from file... ", end='')
            with open(self.dumpfile , 'rb') as f:
                self.satelliteDict = pickle.load(f)
            print("Done.")
            return
        
        for prn in tqdm(self.prnlist, desc="Acquisition progress"):
            acquisition = Acquisition(self.configfile, prn, self.signal)
            acquisition.acquire(self.data_file)
            self.satelliteDict[prn].setAcquisition(acquisition)
        
        # Dump results
        with open(self.dumpfile , 'wb') as f:
            pickle.dump(self.satelliteDict, f, pickle.HIGHEST_PROTOCOL)
        return
    
    # -------------------------------------------------------------------------

    def doTracking(self, loadPrevious=False):
        if loadPrevious:
            print("Tracking progress   : Loading from file... ", end='')
            with open(self.dumpfile , 'rb') as f:
                self.satelliteDict = pickle.load(f)
            print("Done.")
            return
        
        for prn in tqdm(self.prnlist, desc="Tracking progress   "):
            tracking = Tracking(self.configfile, self.satelliteDict[prn].getAcquisition())
            tracking.track(self.data_file)
            self.satelliteDict[prn].setTracking(tracking)
        
        # Dump results
        with open(self.dumpfile , 'wb') as f:
            pickle.dump(self.satelliteDict, f, pickle.HIGHEST_PROTOCOL)
        return
    
    # -------------------------------------------------------------------------

    def doDecoding(self, loadPrevious=False):
        if loadPrevious:
            print("Decoding progress   : Loading from file... ", end='')
            with open(self.dumpfile , 'rb') as f:
                self.satelliteDict = pickle.load(f)
            print("Done.")
            return
        
        for prn in tqdm(self.prnlist, desc="Decoding progress   "):
            decoding = Decoding(self.satelliteDict[prn].getTracking())
            result = decoding.decode()
            # Check for succesful decoding
            if result == 0:
                self.satelliteDict[prn].setDecoding(decoding, True)
            else: 
                self.satelliteDict[prn].setDecoding(decoding, False)
        
        with open(self.dumpfile , 'wb') as f:
            pickle.dump(self.satelliteDict, f, pickle.HIGHEST_PROTOCOL)
        return

    # -------------------------------------------------------------------------

    def doNavigation(self):
        
        # Compute pseudoranges
        self.navigation = Navigation(self.configfile)

        # Check satellite to send for computations
        satdict = self.satelliteDict.copy()
        for prn, satellite in self.satelliteDict.items():
            if not satellite.decodingEnabled:
                satdict.pop(prn)
                continue
        self.navigation.computePseudoranges(satdict)

        return
    

