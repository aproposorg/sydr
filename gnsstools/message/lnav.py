
from warnings import WarningMessage
import numpy as np
import gnsscal
import copy
from gnsstools.ephemeris import BRDCEphemeris
from gnsstools.measurements import DSPmeasurement, TrackingMeasurement
from gnsstools.message.abstract import NavigationMessageAbstract
import gnsstools.constants as constants


class LNAV(NavigationMessageAbstract):

    PREAMBULE_BITS     = np.array([1, 0, 0, 0, 1, 0, 1, 1])
    PREAMBULE_BITS_INV = np.array([0, 1, 1, 1, 0, 1, 0, 0])
    MS_IN_NAV_BIT  = 20 # TODO move to configuration file
    SUBFRAME_BITS  = 300
    WORD_BITS      = 30

    svid           : int

    time           : list

    data           : list
    idxData        : int
    bits           : list
    bitsSamples    : list  # List containing the absolute sample number corresponding each start bit
    idxBits        : int
    bitFound       : bool
    subframeFound  : bool
    idxSubframe    : int
    preambuleFound : bool
    idxPreambule   : int

    bitsLastSubframe : int
    idxFirstSubframe : int   # Current subframe list, used as a buffer when no subframe has been decoded yet.
    subframeDecoded : list
    isBitInverted : bool
    idxLastSubframe : int   # Index to the last decoded subframe, w.r.t. the bit array
    
    # Message contents
    ephemeris        : BRDCEphemeris
    partialEphemeris : BRDCEphemeris
    ura              : float          # User Range Accuracy
    health           : int            # Satellite health flag
    tow              : int
    weekNumber       : int

    # Flags
    isTOWDecoded        : bool
    isEphemerisDecoded  : bool
    isFirstSubframeFound: bool

# -----------------------------------------------------------------------------

    def __init__(self, svid):

        self.svid = svid

        self.data = np.empty(self.MS_IN_NAV_BIT)
        self.partialEphemeris = BRDCEphemeris()

        self.data = np.zeros(self.MS_IN_NAV_BIT)
        self.idxData = 0
        self.time = []
        self.bits = []
        self.bitsSamples = []
        self.idxBits = 0
        self.firstBitFound = False          # Bool to see if a switch of bit has been found, starting point of decoding
        self.bitFound = False               # Track if a new bit is available to be decoded
        self.subframeFound = False
        self.subframeProcessed = False

        self.bitsLastSubframe = 0

        self.tow = 0
        self.weekNumber = 0

        self.isTOWDecoded = False
        self.isEphemerisDecoded = False
        self.isFirstSubframeFound = False
        self.isBitInverted = False

        self.idxFirstSubframe = -1
        self.idxLastSubframe = -1
        self.idxCodeSubframe = -1

        self.codeCounter = 0 # Code counter, reset at each new TOW
        self.idxCode = []

        pass

# -----------------------------------------------------------------------------

    def addMeasurement(self, time, dspMeasurement:TrackingMeasurement):

        self.data[self.idxData] = dspMeasurement.iPrompt
        self.idxData += 1
        self.codeCounter += 1 

        # Check for bit change
        if not self.firstBitFound and self.idxData > 1: 
            if  np.sign(self.data[self.idxData-2]) != np.sign(self.data[self.idxData-1]):
                self.firstBitFound = True
                self.data.fill(0.0)
                self.data[0] = dspMeasurement.iPrompt
                self.idxData = 1

        # Check if enough data is present to decode the bit
        if self.idxData == self.MS_IN_NAV_BIT:
            bit = self.toBits(np.array(self.data), accumulate=self.MS_IN_NAV_BIT)            
            self.bits.append(bit[0])
            self.bitsSamples.append(dspMeasurement.sample)
            self.idxCode.append(dspMeasurement.idx)
            self.data.fill(0.0)
            self.idxData = 0
            self.time.append(time)
            self.bitFound = True
            self.bitsLastSubframe += 1
        
        return

# -----------------------------------------------------------------------------

    def run(self):

        # Check is a new bit is available, otherwise nothing to do
        if not self.bitFound:
            return
        self.bitFound = False
        
        # Check for subframe
        self.checkSubframe()

        # if self.subframeFound and not self.subframeProcessed:
        #     if len(self.bits[self.idxSubframe:]) == self.SUBFRAME_BITS:
        #         self.decodeSubframe()

        return

# -----------------------------------------------------------------------------

    def checkSubframe(self):
        
        # Need at least the preambule bits plus the previous 2 bit to perform checks
        # plus the 2 words afterwards
        minBits = 2 + 2 * self.WORD_BITS
        if len(self.bits) < minBits:
            return 

        # Check if the first subframe has been found
        if not self.isFirstSubframeFound:
            isSecondSubframeFound = False
            idx = len(self.bits) - minBits
            if not self.checkPreambule(idx):
                return
            
            if self.idxFirstSubframe == -1:
                self.idxFirstSubframe = idx
                return
            elif idx - self.idxFirstSubframe != self.SUBFRAME_BITS:
                self.idxFirstSubframe = idx
                return
            
            # The first subframe has been verified
            self.isFirstSubframeFound = True
            
            # Decode first subframe 
            self.decodeSubframe(self.idxFirstSubframe)

            self.bitsLastSubframe = minBits

        if self.bitsLastSubframe < self.SUBFRAME_BITS:
            return
        
        idx = len(self.bits) - self.SUBFRAME_BITS
        if not self.checkPreambule(idx):
            print(f"WARNING: Subframe for satellite G{self.svid} is not found where it was expected. Subframe tracking will be reseted.")
            self.isFirstSubframeFound = False
            self.currSubframeList = []
            return
        
        self.decodeSubframe(idx)
        self.bitsLastSubframe = 0

        # # Need at least the preambule bits plus the previous 2 bit to perform checks
        # # plus the 2 words afterwards
        # minBits = 2 + 2 * self.WORD_BITS
        # if not (len(self.bits) > minBits):
        #     return 

        # # Check subframe if it is not too early for a new subframe
        # if self.subframeFound and len(self.bits[self.idxSubframe:]) < self.SUBFRAME_BITS + minBits:
        #    return

        # self.subframeProcessed = False

        # if (self.bits[-minBits:-minBits+self.PREAMBULE_BITS.size] == self.PREAMBULE_BITS).all() \
        #     or (self.bits[-minBits:-minBits+self.PREAMBULE_BITS_INV.size] == self.PREAMBULE_BITS_INV).all():
        #     idx = len(self.bits) - minBits

        #     # Need to convert the '0' into '-1' for the parity check function
        #     bits = np.array([-1 if x == 0 else 1 for x in self.bits[idx-2:idx+2*self.WORD_BITS]])
        #     if self.parityCheck(bits[:self.WORD_BITS+2]) and \
        #        self.parityCheck(bits[self.WORD_BITS:2*self.WORD_BITS+2]):
        #        self.subframeFound = True
        #        self.idxSubframe = idx
        #        print(f"Subframe found for satellite satellite G{self.svid}.")  
        
        return 

# -----------------------------------------------------------------------------

    def checkPreambule(self, idx):
        subframeFound = False
        if (self.bits[idx:idx+self.PREAMBULE_BITS.size] == self.PREAMBULE_BITS).all() \
            or (self.bits[idx:idx+self.PREAMBULE_BITS.size]  == self.PREAMBULE_BITS_INV).all():

            # Need to convert the '0' into '-1' for the parity check function
            bits = np.array([-1 if x == 0 else 1 for x in self.bits[idx-2:idx+2*self.WORD_BITS]])
            if self.parityCheck(bits[:self.WORD_BITS+2]) and \
               self.parityCheck(bits[self.WORD_BITS:2*self.WORD_BITS+2]):
               subframeFound = True

        return subframeFound


# -----------------------------------------------------------------------------

    def decodeSubframe(self, idxSubframe):
        """
        Decode of the navigation message based on the navigation bits. For
        detailed information about the procedure, refer to the GPS ICD 
        (IS-GPS-200D).
        """

        # Last bit of previous word use to check if bits inversion 
        # is needed.
        d30star = self.bits[idxSubframe - 1]
        subframe = self.bits[idxSubframe : idxSubframe + self.SUBFRAME_BITS]
        
        # Word check 
        for j in range(10):
            subframe[30*j:30*(j+1)] = self.checkPhase(subframe[30*j:30*(j+1)], d30star)
            d30star = subframe[30*(j+1)-1]
    
        # Concatenate the string 
        subframe = ''.join([str(i) for i in subframe])

        subframeID = self.bin2dec(subframe[49:52])

        eph = self.partialEphemeris

        # Identify the subframe
        if subframeID == 1:
            # It contains WN, SV clock corrections, health and accuracy
            self.weekNumber   = self.bin2dec(subframe[60:70]) + constants.GPS_WEEK_ROLLOVER * 1024
            self.accuracy     = self.bin2dec(subframe[72:76])
            self.health       = self.bin2dec(subframe[76:82])
            eph.iodc          = self.bin2dec(subframe[82:84] + subframe[211:218])  # TODO Check IODC consistency
            eph.toc           = self.bin2dec(subframe[218:234]) * 2 ** 4
            eph.tgd           = self.twosComp2dec(subframe[196:204]) * 2 ** (- 31)
            eph.af2           = self.twosComp2dec(subframe[240:248]) * 2 ** (- 55)
            eph.af1           = self.twosComp2dec(subframe[248:264]) * 2 ** (- 43)
            eph.af0           = self.twosComp2dec(subframe[270:292]) * 2 ** (- 31)
            eph.subframe1Flag = True
        elif subframeID == 2:
            # It contains first part of ephemeris parameters
            eph.iode          = self.bin2dec(subframe[60:68]) # TODO Check IODE consistency
            eph.ecc           = self.bin2dec(subframe[166:174] + subframe[180:204]) * 2 ** (- 33)
            eph.sqrtA         = self.bin2dec(subframe[226:234] + subframe[240:264]) * 2 ** (- 19)
            eph.toe           = self.bin2dec(subframe[270:286]) * 2 ** 4
            eph.crs           = self.twosComp2dec(subframe[68:84]) * 2 ** (- 5)
            eph.deltan        = self.twosComp2dec(subframe[90:106]) * 2 ** (- 43) * constants.PI
            eph.m0            = self.twosComp2dec(subframe[106:114] + subframe[120:144]) * 2 ** (- 31) * constants.PI
            eph.cuc           = self.twosComp2dec(subframe[150:166]) * 2 ** (- 29)
            eph.cus           = self.twosComp2dec(subframe[210:226]) * 2 ** (- 29)
            eph.subframe2Flag = True
        elif subframeID == 3:
            # It contains second part of ephemeris parameters
            eph.iode          = self.bin2dec(subframe[270:278]) # TODO Check IODE consistency
            eph.cic           = self.twosComp2dec(subframe[60:76]) * 2 ** (- 29)
            eph.omega0        = self.twosComp2dec(subframe[76:84] + subframe[90:114]) * 2 ** (- 31) * constants.PI
            eph.cis           = self.twosComp2dec(subframe[120:136]) * 2 ** (- 29)
            eph.i0            = self.twosComp2dec(subframe[136:144] + subframe[150:174]) * 2 ** (- 31) * constants.PI
            eph.crc           = self.twosComp2dec(subframe[180:196]) * 2 ** (- 5)
            eph.omega         = self.twosComp2dec(subframe[196:204] + subframe[210:234]) * 2 ** (- 31) * constants.PI
            eph.omegaDot      = self.twosComp2dec(subframe[240:264]) * 2 ** (- 43) * constants.PI
            eph.iDot          = self.twosComp2dec(subframe[278:292]) * 2 ** (- 43) * constants.PI
            eph.subframe3Flag = True

        elif subframeID == 4:
            # Almanac, ionospheric model, UTC parameters.
            # SV health (PRN: 25-32).
            # Not decoded at the moment.
            # TODO
            # self.ephemeris.alpha0 = self.twosComp2dec(subframe[60:76]) * 2 ** (-30)
            # self.ephemeris.alpha1 = self.twosComp2dec(subframe[60:76]) * 2 ** (-27) / constants.PI
            # self.ephemeris.alpha2 = self.twosComp2dec(subframe[60:76]) * 2 ** (-24) / constants.PI**2
            # self.ephemeris.alpha3 = self.twosComp2dec(subframe[60:76]) * 2 ** (-24) / constants.PI**3
            # self.ephemeris.beta0  = self.twosComp2dec(subframe[60:76]) * 2 ** ( 11)
            # self.ephemeris.beta1  = self.twosComp2dec(subframe[60:76]) * 2 ** ( 14) / constants.PI
            # self.ephemeris.beta2  = self.twosComp2dec(subframe[60:76]) * 2 ** ( 16) / constants.PI**2
            # self.ephemeris.beta3  = self.twosComp2dec(subframe[60:76]) * 2 ** ( 16) / constants.PI**3
            pass
        elif subframeID == 5:
            # SV almanac and health (PRN: 1-24).
            # Almanac reference week number and time.
            # Not decoded at the moment.
            # TODO
            pass
        else: 
            print(f"Unrecognised suframe ID {subframeID} found for satellite G{self.svid}")

        if eph.checkFlags():
            self.ephemeris = copy.copy(self.partialEphemeris)
            self.isEphemerisDecoded = True

        # Compute the time of week (TOW) of the first sub-frames in the array ====
        # Also correct the TOW. The transmitted TOW is actual TOW of the next
        # subframe and we need the TOW of the first subframe in this data block
        # (the variable subframe at this point contains bits of the last subframe).
        # Also the TOW written in the message is referred to very begining of the 
        # subframe, meaning the first bit of the preambule.
        # So we remove 6 seconds to have the TOW of the current subframe
        self.tow = self.bin2dec(subframe[30:47]) * 6 - 6
        self.idxLastSubframe = idxSubframe
        self.isTOWDecoded = True
        self.codeCounter = 0

        if self.tow != 0 and self.weekNumber != 0:
            self.doy = gnsscal.gpswd2yrdoy(self.weekNumber, \
                                            int(self.tow / constants.SECONDS_PER_DAY))

        self.subframeProcessed = True

        print(f"Subframe {subframeID} decoded for satellite satellite G{self.svid}.") 

        return
    
    # -------------------------------------------------------------------------

    def getSampleSubframe(self):
        """
        Get the sample number of the last TOW.
        """
        return self.bitsSamples[self.idxLastSubframe]

    # -------------------------------------------------------------------------

    def getCodeSubframe(self):
        """
        Get the index of the code of the last subframe.
        """

        return self.idxCode[self.idxLastSubframe]


    # -------------------------------------------------------------------------
