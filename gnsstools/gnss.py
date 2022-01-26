import configparser
from gnsstools.gnsssignal import GNSSSignal
from gnsstools.acquisition import Acquisition
from gnsstools.rffile import RFFile

class GNSS:
    def __init__(self, configfile):
        # Initialise from config file
        config = configparser.ConfigParser()
        config.read(configfile)
    	
        self.configfile = configfile
        self.signalfile = config.get('DEFAULT', 'signalpath')
        
        self.signal_type = config.get('DEFAULT', 'signal_type')

        # Instanciate variables
        self.data_file = RFFile(self.configfile)
        self.signal = GNSSSignal(self.signalfile, self.signal_type)
        self.acquisition = Acquisition(configfile)

        return

    def doAcquisition(self, prnlist):
        
        print( "+-----+--------+----------+----------+")
        print(f"| PRN | METRIC | DOPPLER  | CODE     |")
        print( "+-----+--------+----------+----------+")
        for prn in prnlist:
            [acq_corr, acq_metric, coarse_freq, coarse_code] = self.acquisition.acquire(self.data_file, prn, self.signal)
            print(f"| G{prn:02} | {acq_metric:>6.2f} | {coarse_freq:>8.2f} | {coarse_code:>8.2f} |")
        print( "+-----+--------+----------+----------+")

        return

