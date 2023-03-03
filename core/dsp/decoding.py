
import numpy as np
from core.utils.constants import LNAV_PREAMBULE_SIZE, LNAV_PREAMBULE_BITS, LNAV_PREAMBULE_BITS_INV, LNAV_WORD_SIZE

# =====================================================================================================================

def Prompt2Bit(prompt:float, bit0:int=0):
    """
    Convert prompt correlator results to binary vector.

    Args:
        prompt (float): prompt result to be converted.
        bit0 (integer): Representation of the 0 bit, like 0 or -1
    
    Returns
        bit (int): Binary bit result
    """
    return 1 if prompt > 0 else bit0

# =====================================================================================================================

def ParityCheck(ndat):
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

# =====================================================================================================================

def LNAV_DecodeSubframe():
    
    return

# =====================================================================================================================

def LNAV_CheckPreambule(bits:np.array):
    """
    Check bits for preambule. The bits array (0/1), starting from the previous paritiy check bits and first two 
    words bits and parity bits. For example, if i is the index of presumed start of the preambule, the array should be 
    provided as:
        bits [ i - 2 : i + 2 * WORD_SIZE + 2]
    WORD_SIZE being 30 for LNAV messages.
    See reference [GPS ICD].

    Args:
        bits (np.array): Binary array (0/1).
    
    Returns
        subframeFound (bool): Success bool.

    Raises:
        None
    """
    # TODO Could do bitwise operations instead of table comparison?
    
    subframeFound = False
    if (bits[2:LNAV_PREAMBULE_SIZE] == LNAV_PREAMBULE_BITS).all() \
        or (bits[2:LNAV_PREAMBULE_SIZE] == LNAV_PREAMBULE_BITS_INV).all():

        # Need to convert the '0' into '-1' for the parity check function
        convertedBits = np.array([-1 if x == 0 else 1 for x in bits[:2*LNAV_WORD_SIZE]])
        
        if ParityCheck(convertedBits[:LNAV_WORD_SIZE+2]) and \
            ParityCheck(convertedBits[LNAV_WORD_SIZE:2*LNAV_WORD_SIZE+2]):
            subframeFound = True
    
    return subframeFound


# =====================================================================================================================
