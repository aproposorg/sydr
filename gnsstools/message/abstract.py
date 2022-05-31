
import numpy as np

from enum import Enum, unique
# =============================================================================
@unique
class NavMessageType(Enum):
    GPS_LNAV = 0

# =============================================================================
class NavigationMessageAbstract():
    
    navMessageType : NavMessageType

    towFound : bool

    def __init__(self):

        return

    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------

    @staticmethod
    def parityCheck(ndat):
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
    def toBits(data, accumulate=1, bit0=0):
        """
        Convert signal to binary vector.

        Inputs
            data : np.array
                data to be converted.
            accumulate : integer
                Number of bits to accumulate for noisy signals
            bit0 : integer
                Representation of the 0 bit, like 0 or -1
        Outputs
            bits : np.array
                Binary vector
        """
        # Reshape in as many arrays  as to accumulate
        bits = data.reshape(accumulate, -1, order='F')
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
        if d30star == 1:
            # Data bits must be inverted
            for i in range(0, 24):
                if word[i] == 1:
                    word[i] = 0
                elif word[i] == 0:
                    word[i] = 1
        return word

# =============================================================================