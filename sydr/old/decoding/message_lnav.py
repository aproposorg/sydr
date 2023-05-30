
import logging
import numpy as np

from sydr.satellite.ephemeris import BRDCEphemeris
from sydr.decoding.message_abstract import NavigationMessageAbstract, MessageType
import sydr.utils.constants as constants

class LNAV(NavigationMessageAbstract):

    PREAMBULE_BITS     = np.array([1, 0, 0, 0, 1, 0, 1, 1])
    PREAMBULE_BITS_INV = np.array([0, 1, 1, 1, 0, 1, 0, 0])
    MS_IN_NAV_BIT      = 20 # TODO move to configuration file
    SUBFRAME_BITS      = 300
    WORD_BITS          = 30

    svid           : int
    data           : list # Output from prompt correlator, reset at each new bit.
    idxData        : int  # Current index of data table, reset at each new bit.
    bits           : list # List of bits decoded.
    bitsSamples    : list # List containing the absolute sample number corresponding each start bit
    TOWInSamples   : int 

    bitsLastSubframe : int # Number of bits since last subframe.
    idxFirstSubframe : int # Current subframe list, used as a buffer when no subframe has been decoded yet.
    idxLastSubframe  : int # Index to the last decoded subframe, w.r.t. the bit array.
    
    # Message contents
    ephemeris        : BRDCEphemeris
    tow              : int

    # Flags
    isTOWDecoded        : bool # Track if the TOW has been decoded.
    isEphemerisDecoded  : bool # Track if subframe 1,2,3 have been decoded.
    isFirstBitFound     : bool # Track if a switch of bit has been found, starting point of decoding.
    isNewBitFound       : bool # Track if a new bit is available to be decoded.
    isFirstSubframeFound: bool # Track if at least one subframe has been found.

    subframes : dict

# -----------------------------------------------------------------------------

    def __init__(self):

        self.type = MessageType.GPS_LNAV

        # Initialise objects
        self.data             = np.zeros(self.MS_IN_NAV_BIT)
        self.idxData          = 0
        self.bits             = []
        self.bitsSamples      = []
        self.bitsLastSubframe = 0 
        self.TOWInSamples     = -1

        self.idxFirstSubframe = -1
        self.idxLastSubframe  = -1

        # Flags
        self.isTOWDecoded         = False
        self.isEphemerisDecoded   = False
        self.isFirstBitFound      = False
        self.isNewBitFound        = False
        self.isNewSubframeFound   = False 
        self.isFirstSubframeFound = False
        
        # Data
        self.tow = 0
        self.ephemeris = BRDCEphemeris()
        self.lastSubframeID = -1 
        self.lastSubframeBits = None

        self.subframes = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0} # Contains the IODE of the lastest subframe received

        pass

# -----------------------------------------------------------------------------

    def addMeasurement(self, timeInSamples, iPrompt):

        self.data[self.idxData] = iPrompt
        self.idxData += 1

        # Check for bit change
        if not self.isFirstBitFound and self.idxData > 1: 
            if  np.sign(self.data[self.idxData-2]) != np.sign(self.data[self.idxData-1]):
                self.isFirstBitFound = True
                self.data.fill(0.0)
                self.data[0] = iPrompt
                self.idxData = 1

        # Check if enough data is present to decode the bit
        if self.idxData == self.MS_IN_NAV_BIT:
            bit = self.toBits(np.array(self.data), accumulate=self.MS_IN_NAV_BIT)            
            self.bits.append(bit[0])
            self.bitsSamples.append(timeInSamples)
            self.data.fill(0.0)
            self.idxData = 0
            self.isNewBitFound = True
            self.bitsLastSubframe += 1
        
        return

# -----------------------------------------------------------------------------

    def run(self):

        # Check is a new bit is available, otherwise nothing to do
        if not self.isNewBitFound:
            return
        self.isNewBitFound = False
        
        # Check for subframe
        self.checkSubframe()

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
            idx = len(self.bits) - minBits
            # Check if the preambule is in the last bits decoded
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
            self.bitsLastSubframe -= self.idxFirstSubframe
            
            # Decode first subframe 
            self.decodeSubframe(self.idxFirstSubframe)

            self.bitsLastSubframe = minBits

        # If found, just check the number of bits since last frame instead
        if self.bitsLastSubframe < self.SUBFRAME_BITS:
            return
        
        idx = len(self.bits) - self.SUBFRAME_BITS
        if not self.checkPreambule(idx):
            return
        
        self.decodeSubframe(idx)
        self.bitsLastSubframe = 0
        
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
        self.lastSubframeID = subframeID
        self.lastSubframeBits = subframe

        eph = self.ephemeris
        # Identify the subframe
        if subframeID == 1:
            # It contains WN, SV clock corrections, health and accuracy
            eph.weekNumber    = self.bin2dec(subframe[60:70]) + constants.GPS_WEEK_ROLLOVER * 1024
            eph.ura           = self.bin2dec(subframe[72:76])
            eph.health        = self.bin2dec(subframe[76:82])
            eph.iodc          = self.bin2dec(subframe[82:84] + subframe[211:218])  # TODO Check IODC consistency
            eph.toc           = self.bin2dec(subframe[218:234]) * 2 ** 4
            eph.tgd           = self.twosComp2dec(subframe[196:204]) * 2 ** (- 31)
            eph.af2           = self.twosComp2dec(subframe[240:248]) * 2 ** (- 55)
            eph.af1           = self.twosComp2dec(subframe[248:264]) * 2 ** (- 43)
            eph.af0           = self.twosComp2dec(subframe[270:292]) * 2 ** (- 31)
            eph.subframe1Flag = True
            self.weekNumber   = eph.weekNumber
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
        
        self.subframes[subframeID] = 1

        if eph.checkFlags():
            self.isEphemerisDecoded = True
        
        self.isNewSubframeFound = True
        self.idxLastSubframe = idxSubframe

        # Actualize TOW
        # Compute the time of week (TOW) of the first sub-frames in the array
        # - The transmitted TOW is actual TOW of the next subframe and we need the TOW of the first subframe in this data block
        # - The TOW written in the message is referred to very begining of the subframe, meaning the first bit of the preambule.
        # -> So we remove 6 seconds to have the TOW of the current subframe
        # - Also need to realign the TOW to the current process time 
        # -> account for all the bits since last subframe was decoded
        tow  = self.bin2dec(subframe[30:47]) * 6 
        tow -= 6
        self.tow = tow
        eph.tow  = self.tow
        
        # Correct TOW to actual state
        # TODO Should we make a function instead since this is not the pure TOW contained in the message
        self.tow += self.bitsLastSubframe * self.MS_IN_NAV_BIT * 1e-3

        self.TOWInSamples = self.bitsSamples[idxSubframe]
        self.isTOWDecoded = True

        # if self.tow != 0 and self.weekNumber != 0:
        #     self.doy = gnsscal.gpswd2yrdoy(self.weekNumber, \
        #                                     int(self.tow / constants.SECONDS_PER_DAY))
        
        #print(f"Subframe {subframeID} decoded for satellite G{self.svid} (TOW = {eph.tow}).") 
        logging.getLogger(__name__).info(f"Subframe {subframeID} decoded for satellite G{self.svid} (TOW = {eph.tow}).")

        return
    
    # -------------------------------------------------------------------------

    def getDatabaseDict(self):
        """
        Contains the information to be save in the database in the form of a 
        dictionnary. The key is the column name.

        Returns:
            mdict (Dict): Information to be saved.

        """
        
        mdict = {
            "tow"        : self.ephemeris.tow, 
            "subframe_id" : self.lastSubframeID,
            #"bits"       : self.bits[self.idxLastSubframe : self.idxLastSubframe + self.SUBFRAME_BITS]
            "bits"       : self.lastSubframeBits
        }

        return mdict


    # -------------------------------------------------------------------------
