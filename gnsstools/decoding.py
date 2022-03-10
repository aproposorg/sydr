import numpy as np
from gnsstools.ephemeris import Ephemeris
import gnsstools.constants as constants
from gnsstools.tracking import Tracking
import gnsscal

class Decoding:

    PREAMBULE_BITS = np.array([1, -1, -1, -1, 1, -1, 1, 1])
    MS_IN_NAV_BIT = 20 # TODO move to configuration file
    MS_IN_SUBFRAME = 6000

    PI = 3.1415926535898 # Pi number used for GPS TODO move to configuration file

    # -------------------------------------------------------------------------

    def __init__(self, track:Tracking):

        self.track = track
        self.msToProcess = track.msToProcess
        self.dataSignal = track.iPrompt.copy()
        self.prn = track.prn
        self.firstSubFrame = np.NaN
        self.isActive = False

        self.ephemeris = Ephemeris()

        return

    # -------------------------------------------------------------------------

    def decode(self):

        # Find the preambule in the signal
        self.findPreambles(self.track.iPrompt)
        if not self.isActive:
            print(f"Could not find preambule in satellite tracked {self.prn}.")
            return -1

        # Decode ephemeris
        self.decodeNavigationBits()
        
        return 0

    # -------------------------------------------------------------------------

    def decodeNavigationBits(self):
        """
        Decode of the navigation message based on the navigation bits. For
        detailed information about the procedure, refer to the GPS ICD 
        (IS-GPS-200D).
        """

        # Select the subframes
        # TODO could do the accumulation earlier and save the results in the class variables
        ## Take the last 1 bits in previous subframe
        idx_start = self.firstSubFrame - (1*self.MS_IN_NAV_BIT)
        ## Take the 5 subframes (each subframe is 300 bits)
        idx_stop  = self.firstSubFrame + (300 * 5 * self.MS_IN_NAV_BIT) 

        signal = self.dataSignal[idx_start:idx_stop]
        bits = self.toBits(signal, accumulate=self.MS_IN_NAV_BIT)
        bits = list(map(str, bits)) # Map to string
        
        # Last bit of previous word use to check if bits inversion 
        # is needed.
        d30star = bits[0] 
        bits = bits[1:]
        
        # Decode subframes
        ## TODO Change hardcoded values to constant related to the signal
        for i in range(5):
            subframe = bits[300*i:300*(i+1)]
            for j in range(10):
                subframe[30*j:30*(j+1)] = self.checkPhase(subframe[30*j:30*(j+1)], d30star)
                d30star = subframe[30*(j+1)-1]
        
            ## Concatenate the string 
            subframe = ''.join(subframe)
            subframeID = self.bin2dec(subframe[49:52])

            ## Identify the subframe
            if subframeID == 1:
                # It contains WN, SV clock corrections, health and accuracy
                self.ephemeris.weekNumber = self.bin2dec(subframe[60:70]) + constants.GPS_WEEK_ROLLOVER * 1024
                self.ephemeris.accuracy   = self.bin2dec(subframe[72:76])
                self.ephemeris.health     = self.bin2dec(subframe[76:82])
                self.ephemeris.IODC       = self.bin2dec(subframe[82:84] + subframe[211:218])
                self.ephemeris.t_oc       = self.bin2dec(subframe[218:234]) * 2 ** 4
                self.ephemeris.T_GD       = self.twosComp2dec(subframe[196:204]) * 2 ** (- 31)
                self.ephemeris.a_f2       = self.twosComp2dec(subframe[240:248]) * 2 ** (- 55)
                self.ephemeris.a_f1       = self.twosComp2dec(subframe[248:264]) * 2 ** (- 43)
                self.ephemeris.a_f0       = self.twosComp2dec(subframe[270:292]) * 2 ** (- 31)
            elif 2 == subframeID:
                # It contains first part of ephemeris parameters
                self.ephemeris.IODE_sf2   = self.bin2dec(subframe[60:68])
                self.ephemeris.e          = self.bin2dec(subframe[166:174] + subframe[180:204]) * 2 ** (- 33)
                self.ephemeris.sqrtA      = self.bin2dec(subframe[226:234] + subframe[240:264]) * 2 ** (- 19)
                self.ephemeris.t_oe       = self.bin2dec(subframe[270:286]) * 2 ** 4
                self.ephemeris.C_rs       = self.twosComp2dec(subframe[68:84]) * 2 ** (- 5)
                self.ephemeris.deltan     = self.twosComp2dec(subframe[90:106]) * 2 ** (- 43) * self.PI
                self.ephemeris.M_0        = self.twosComp2dec(subframe[106:114] + subframe[120:144]) * 2 ** (- 31) * self.PI
                self.ephemeris.C_uc       = self.twosComp2dec(subframe[150:166]) * 2 ** (- 29)
                self.ephemeris.C_us       = self.twosComp2dec(subframe[210:226]) * 2 ** (- 29)
            elif 3 == subframeID:
                # It contains second part of ephemeris parameters
                self.ephemeris.C_ic       = self.twosComp2dec(subframe[60:76]) * 2 ** (- 29)
                self.ephemeris.omega_0    = self.twosComp2dec(subframe[76:84] + subframe[90:114]) * 2 ** (- 31) * self.PI
                self.ephemeris.C_is       = self.twosComp2dec(subframe[120:136]) * 2 ** (- 29)
                self.ephemeris.i_0        = self.twosComp2dec(subframe[136:144] + subframe[150:174]) * 2 ** (- 31) * self.PI
                self.ephemeris.C_rc       = self.twosComp2dec(subframe[180:196]) * 2 ** (- 5)
                self.ephemeris.omega      = self.twosComp2dec(subframe[196:204] + subframe[210:234]) * 2 ** (- 31) * self.PI
                self.ephemeris.omegaDot   = self.twosComp2dec(subframe[240:264]) * 2 ** (- 43) * self.PI
                self.ephemeris.iDot       = self.twosComp2dec(subframe[278:292]) * 2 ** (- 43) * self.PI
                self.ephemeris.IODE_sf3   = self.bin2dec(subframe[270:278])

            elif 4 == subframeID:
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
            elif 5 == subframeID:
                # SV almanac and health (PRN: 1-24).
                # Almanac reference week number and time.
                # Not decoded at the moment.
                # TODO
                pass

        # Compute the time of week (TOW) of the first sub-frames in the array ====
        # Also correct the TOW. The transmitted TOW is actual TOW of the next
        # subframe and we need the TOW of the first subframe in this data block
        # (the variable subframe at this point contains bits of the last subframe).
        # Also the TOW written in the message is referred to very begining of the 
        # subframe, meaning the first bit of the preambule.
        self.TOW = self.bin2dec(subframe[30:47]) * 6 - 30

        if self.TOW and self.ephemeris.weekNumber:
            self.doy = gnsscal.gpswd2yrdoy(self.ephemeris.weekNumber, \
                                           int(self.TOW / constants.SECONDS_PER_DAY))

        return
    
    # -------------------------------------------------------------------------

    def findPreambles(self, signal):
        preambule = np.kron(self.PREAMBULE_BITS, np.ones(20))

        # TODO move the conversion to bit in tracking results directly
        iPromptBits = np.zeros_like(signal)
        iPromptBits[signal >  0] =  1
        iPromptBits[signal <= 0] = -1

        correlation = np.correlate(iPromptBits, np.pad(preambule, (0, iPromptBits.size - preambule.size), 'constant'), mode='full')

        # Take only what we want, the first part is due to 0 padding
        # Every correlation results below 153 is probably not the preambule
        # Maybe should use the max value instead? 153 feels very hardcoded
        size = int((len(correlation) + 1) / 2)
        index = (np.abs(correlation[size-1:size*2]) > 153).nonzero()[0] 

        # Analyse the premabules 
        for i in index:
            # Pre-check based on sub-frame length
            if not (index - i == self.MS_IN_SUBFRAME).any():
                continue
            # Since each bit is 20 ms, there 20 values available for each bit.
            # To help decoding in noisy signal, all these bits are accumulated.
            # 62 bits taken in total for parity check
            # Take the last 2 bits in previous subframe
            idx_start = i - (2*self.MS_IN_NAV_BIT)
            # Take the first two 30 bit words (TLM + HOW)
            idx_stop  = i + (30 * 2 * self.MS_IN_NAV_BIT) 

            bits = signal[idx_start:idx_stop].copy()
            # Reshape in 20 arrays of 62 bits
            bits = bits.reshape(20, -1, order='F')
            # Accumulate the results
            bits = bits.sum(0)
            bits[bits >  0] =  1
            bits[bits <= 0] = -1

            if self.navPartyChk(bits[:32]) != 0 and self.navPartyChk(bits[30:62]) !=0:
                self.firstSubFrame = i
                self.isActive = True
                break

        return 

    # -------------------------------------------------------------------------

    @staticmethod
    def navPartyChk(ndat):
        """
        From [Borre, 2007] and SoftGNSS-Python (Github)
        This function is called to compute and status the parity bits on GPS word.
        Based on the flowchart in Figure 2-10 in the 2nd Edition of the GPS-SPS
        Signal Spec.

        status = navPartyChk(ndat)

          Inputs:
              ndat        - an array (1x32) of 32 bits represent a GPS navigation
                          word which is 30 bits plus two previous bits used in
                          the parity calculation (-2 -1 0 1 2 ... 28 29)

          Outputs:
              status      - the test value which equals EITHER +1 or -1 if parity
                          PASSED or 0 if parity fails.  The +1 means bits #1-24
                          of the current word have the correct polarity, while -1
                          means the bits #1-24 of the current word must be
                          inverted.

        In order to accomplish the exclusive or operation using multiplication
        this program represents a '0' with a '-1' and a '1' with a '1' so that
        the exclusive or table holds true for common data operations

        	a	b	xor 			a	b	product
         --------------          -----------------
        	0	0	 1			   -1  -1	   1
        	0	1	 0			   -1   1	  -1
        	1	0	 0			    1  -1	  -1
        	1	1	 1			    1   1	   1
        """
        

        # --- Check if the data bits must be inverted ------------------------------
        if ndat[1] != 1:
            ndat[2:26] *= (-1)

        # --- Calculate 6 parity bits ----------------------------------------------
        # The elements of the ndat array correspond to the bits showed in the table
        # 20-XIV (ICD-200C document) in the following way:
        # The first element in the ndat is the D29* bit and the second - D30*.
        # The elements 3 - 26 are bits d1-d24 in the table.
        # The elements 27 - 32 in the ndat array are the received bits D25-D30.
        # The array "parity" contains the computed D25-D30 (parity) bits.
        parity = np.zeros(6)
        parity[0] = ndat[0] * ndat[2] * ndat[3] * ndat[4] * ndat[6] * \
                    ndat[7] * ndat[11] * ndat[12] * ndat[13] * ndat[14] * \
                    ndat[15] * ndat[18] * ndat[19] * ndat[21] * ndat[24]

        parity[1] = ndat[1] * ndat[3] * ndat[4] * ndat[5] * ndat[7] * \
                    ndat[8] * ndat[12] * ndat[13] * ndat[14] * ndat[15] * \
                    ndat[16] * ndat[19] * ndat[20] * ndat[22] * ndat[25]

        parity[2] = ndat[0] * ndat[2] * ndat[4] * ndat[5] * ndat[6] * \
                    ndat[8] * ndat[9] * ndat[13] * ndat[14] * ndat[15] * \
                    ndat[16] * ndat[17] * ndat[20] * ndat[21] * ndat[23]

        parity[3] = ndat[1] * ndat[3] * ndat[5] * ndat[6] * ndat[7] * \
                    ndat[9] * ndat[10] * ndat[14] * ndat[15] * ndat[16] * \
                    ndat[17] * ndat[18] * ndat[21] * ndat[22] * ndat[24]

        parity[4] = ndat[1] * ndat[2] * ndat[4] * ndat[6] * ndat[7] * \
                    ndat[8] * ndat[10] * ndat[11] * ndat[15] * ndat[16] * \
                    ndat[17] * ndat[18] * ndat[19] * ndat[22] * ndat[23] * \
                    ndat[25]

        parity[5] = ndat[0] * ndat[4] * ndat[6] * ndat[7] * ndat[9] * \
                    ndat[10] * ndat[11] * ndat[12] * ndat[14] * ndat[16] * \
                    ndat[20] * ndat[23] * ndat[24] * ndat[25]

        # --- Compare if the received parity is equal the calculated parity --------
        if (parity == ndat[26:]).sum() == 6:
            # Parity is OK. Function output is -1 or 1 depending if the data bits
            # must be inverted or not. The "ndat[2]" is D30* bit - the last  bit of
            # previous subframe.
            status = -1 * ndat[1]

        else:
            # Parity failure
            status = 0

        return status
    
    # -------------------------------------------------------------------------

    @staticmethod
    def toBits(signal, accumulate=1, bit0=0):
        """
        Convert signal to binary vector.

        Inputs
            signal : np.array
                Signal to be converted.
            accumulate : integer
                Number of bits to accumulate for noisy signals
            bit0 : integer
                Representation of the 0 bit, like 0 or -1
        Outputs
            bits : np.array
                Binary vector
        """
        # Reshape in as many arrays  as to accumulate
        bits = signal.reshape(accumulate, -1, order='F')
        # Accumulate the results
        bits = bits.sum(0)
        bits[bits >  0] = 1
        bits[bits <= 0] = bit0

        bits = np.array(bits, dtype=np.int8)

        return bits
   
    # -------------------------------------------------------------------------

    @staticmethod
    def bin2dec(binaryStr):
        assert isinstance(binaryStr, str)
        return int(binaryStr, 2)

    # -------------------------------------------------------------------------
    
    @staticmethod
    def twosComp2dec(binaryStr):
        # TWOSCOMP2DEC(binaryNumber) Converts a two's-complement binary number
        # BINNUMBER (in Matlab it is a string type), represented as a row vector of
        # zeros and ones, to an integer.

        # intNumber = twosComp2dec(binaryNumber)

        # --- Check if the input is string ------------------------------------
        if not isinstance(binaryStr, str):
            raise IOError('Input must be a string.')

        # --- Convert from binary form to a decimal number --------------------
        intNumber = int(binaryStr, 2)

        # --- If the number was negative, then correct the result -------------
        if binaryStr[0] == '1':
            intNumber -= 2 ** len(binaryStr)
        
        return intNumber

    # -------------------------------------------------------------------------
    
    @staticmethod
    def checkPhase(word, d30star):
        # Checks the parity of the supplied 30bit word.
        # The last parity bit of the previous word is used for the calculation.
        # A note on the procedure is supplied by the GPS standard positioning
        # service signal specification.

        # word = checkPhase(word, D30Star)

        #   Inputs:
        #       word        - an array with 30 bit long word from the navigation
        #                   message (a character array, must contain only '0' or
        #                   '1').
        #       D30Star     - the last bit of the previous word (char type).

        #   Outputs:
        #       word        - word with corrected polarity of the data bits
        #                   (character array).

        word_new = []
        if d30star == '1':
            # Data bits must be inverted
            for i in range(0, 24):
                if word[i] == '1':
                    word[i] = '0'
                elif word[i] == '0':
                    word[i] = '1'
        return word