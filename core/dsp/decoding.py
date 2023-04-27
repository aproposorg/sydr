
import numpy as np
from enum import Enum, unique

from core.space.ephemeris import BRDCEphemeris
from core.utils.constants import PI, GPS_WEEK_ROLLOVER, \
    LNAV_PREAMBULE_SIZE, LNAV_PREAMBULE_BITS, LNAV_PREAMBULE_BITS_INV, LNAV_WORD_SIZE

# =============================================================================
@unique
class MessageType(Enum):
    GPS_LNAV = 0

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

def bin2dec(binaryStr):
    """
    Convert binary string to an integer.

    Args:
        binaryStr (np.array): Binary array (0/1).
    
    Returns
        intNumber (int): Converted integer number.

    Raises:
        None
    """
    assert isinstance(binaryStr, str)
    return int(binaryStr, 2)

# =====================================================================================================================

def twosComp2dec(binaryStr):
    """
    Converts a two's-complement binary number to an integer.
    Taken from [Borre, 2007] and [SoftGNSS-Python].

    Args:
        binaryStr (np.array): Binary array (0/1).
    
    Returns
        intNumber (int): Converted integer number.

    Raises:
        IOError: Input binary string must be an str object.
    """

    # Check if the input is string
    if not isinstance(binaryStr, str):
        raise IOError('Input must be a string.')

    # Convert from binary form to a decimal number 
    intNumber = int(binaryStr, 2)

    # If the number was negative, then correct the result
    if binaryStr[0] == '1':
        intNumber -= 2 ** len(binaryStr)
    
    return intNumber

# =============================================================================

def phaseCheck(word, d30star):
    """
    Checks the parity of the supplied 30bit word.
    The last parity bit of the previous word is used for the calculation.
    A note on the procedure is supplied by the GPS standard positioning
    service signal specification.

    word = phaseCheck(word, D30Star)

      Inputs:
          word        - an array with 30 bit long word from the navigation
                      message (a character array, must contain only '0' or
                      '1').
          D30Star     - the last bit of the previous word (char type).

      Outputs:
          word        - word with corrected polarity of the data bits
                      (character array).

    """

    if d30star == 1:
        # Data bits must be inverted
        for i in range(0, 24):
            if word[i] == 1:
                word[i] = 0
            elif word[i] == 0:
                word[i] = 1
    return word

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
    if all(parity == ndat[26:]):
        # Parity is OK. Function output is -1 or 1 depending if the data bits
        # must be inverted or not. The "ndat[2]" is D30* bit - the last  bit of
        # previous subframe.
        status = -1 * ndat[1]

    else:
        # Parity failure
        status = 0

    return status

# =====================================================================================================================

def LNAV_WordsCheck(subframeBits:np.array, d30star:int):
    """
    Check the paritiy of the 10 words in the subframe, and inverse if necessary.

    Args:
            subframeBits (np.array): Binary array (0/1) of size SUBFRAME_SIZE.
            d30star (int): Bit (0/1) for inversion check.
    
    Returns
        subframeBits (np.array): Corrected binary array.

    Raises:
        None

    """
    for j in range(10):
        subframeBits[30*j:30*(j+1)] = phaseCheck(subframeBits[30*j:30*(j+1)], d30star)
        d30star = subframeBits[30*(j+1)-1]

    return subframeBits

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
    if all(bits[2:2+LNAV_PREAMBULE_SIZE] == LNAV_PREAMBULE_BITS) \
        or all(bits[2:2+LNAV_PREAMBULE_SIZE] == LNAV_PREAMBULE_BITS_INV):

        # Need to convert the '0' into '-1' for the parity check function
        convertedBits = np.array([-1 if x == 0 else 1 for x in bits])
        
        if ParityCheck(convertedBits[:LNAV_WORD_SIZE+2]) and \
            ParityCheck(convertedBits[LNAV_WORD_SIZE:2*LNAV_WORD_SIZE+2]):
            subframeFound = True
    
    return subframeFound

# =====================================================================================================================

def LNAV_DecodeTOW(subframeBits:np.array, d30star:int):
    """
    Decode only the TOW from the navigation message, based on the navigation bits. The bits array (0/1) should start 
    from the first subframe bit and the d30star argument for bit inversion is provided as the second argument. 
    For detailed information about the procedure, see reference [GPS ICD].

    Args:
        subframeBits (np.array): Binary array (0/1) of size SUBFRAME_SIZE.
        d30star (int): Bit (0/1) for inversion check.
    
    Returns
        tow (int): Time of Week, non-corrected.
        subframeID (int): Subframe ID.
        subframeBitsStr (string): Subframe bits concatenated into a string

    Raises:
        None
    
    """

    # Words check 
    subframeBits = LNAV_WordsCheck(subframeBits, d30star)

    # Concatenate the string 
    subframeBitsStr = ''.join([str(i) for i in subframeBits])

    # Decode
    subframeID = bin2dec(subframeBitsStr[49:52])
    
    tow  = bin2dec(subframeBitsStr[30:47]) * 6 
    #tow -= 6 # Remove 6 seconds to correct to current subframe (see LNAV_DecodeSubframe)

    return tow, subframeID, subframeBitsStr

# =====================================================================================================================

def LNAV_DecodeSubframe(subframeBits:np.array, d30star:int, ephemeris:BRDCEphemeris):
    """
    Decode of the navigation message based on the navigation bits. The bits array (0/1) should start from the first
    subframe bit and the d30star argument for bit inversion is provided as the second argument. For detailed 
    information about the procedure, see reference [GPS ICD].

    Args:
        subframeBits (np.array): Binary array (0/1) of size SUBFRAME_SIZE.
        d30star (int): Bit (0/1) for inversion check.
        ephemeris (BRDCEphemeris): Ephemeris to store the decoded the message.
    
    Returns
        tow (int): Time of Week, corrected to current subframe.
        ephemeris (BRDCEphemeris): Updated ephemeris based on the one provided.

    Raises:
        None
    
    """
    
    # Words check 
    LNAV_WordsCheck(subframeBits, d30star)

    # Concatenate the string 
    # TODO A bit ugly but the best I could find in Python. Maybe need to switch to bitstream one day?
    subframeBits = ''.join([str(i) for i in subframeBits])

    subframeID = bin2dec(subframeBits[49:52])
    
    # Identify the subframe
    if subframeID == 1:
        # It contains WN, SV clock corrections, health and accuracy
        ephemeris.week          = bin2dec(subframeBits[60:70]) + GPS_WEEK_ROLLOVER * 1024
        ephemeris.ura           = bin2dec(subframeBits[72:76])
        ephemeris.health        = bin2dec(subframeBits[76:82])
        ephemeris.iodc          = bin2dec(subframeBits[82:84] + subframeBits[211:218])  # TODO Check IODC consistency
        ephemeris.toc           = bin2dec(subframeBits[218:234]) * 2 ** 4
        ephemeris.tgd           = twosComp2dec(subframeBits[196:204]) * 2 ** (- 31)
        ephemeris.af2           = twosComp2dec(subframeBits[240:248]) * 2 ** (- 55)
        ephemeris.af1           = twosComp2dec(subframeBits[248:264]) * 2 ** (- 43)
        ephemeris.af0           = twosComp2dec(subframeBits[270:292]) * 2 ** (- 31)
        ephemeris.subframe1Flag = True
    elif subframeID == 2:
        # It contains first part of ephemeris parameters
        ephemeris.iode          = bin2dec(subframeBits[60:68]) # TODO Check IODE consistency
        ephemeris.ecc           = bin2dec(subframeBits[166:174] + subframeBits[180:204]) * 2 ** (- 33)
        ephemeris.sqrtA         = bin2dec(subframeBits[226:234] + subframeBits[240:264]) * 2 ** (- 19)
        ephemeris.toe           = bin2dec(subframeBits[270:286]) * 2 ** 4
        ephemeris.crs           = twosComp2dec(subframeBits[68:84]) * 2 ** (- 5)
        ephemeris.deltan        = twosComp2dec(subframeBits[90:106]) * 2 ** (- 43) * PI
        ephemeris.m0            = twosComp2dec(subframeBits[106:114] + subframeBits[120:144]) * 2 ** (- 31) * PI
        ephemeris.cuc           = twosComp2dec(subframeBits[150:166]) * 2 ** (- 29)
        ephemeris.cus           = twosComp2dec(subframeBits[210:226]) * 2 ** (- 29)
        ephemeris.subframe2Flag = True
    elif subframeID == 3:
        # It contains second part of ephemeris parameters
        ephemeris.iode          = bin2dec(subframeBits[270:278]) # TODO Check IODE consistency
        ephemeris.cic           = twosComp2dec(subframeBits[60:76]) * 2 ** (- 29)
        ephemeris.omega0        = twosComp2dec(subframeBits[76:84] + subframeBits[90:114]) * 2 ** (- 31) * PI
        ephemeris.cis           = twosComp2dec(subframeBits[120:136]) * 2 ** (- 29)
        ephemeris.i0            = twosComp2dec(subframeBits[136:144] + subframeBits[150:174]) * 2 ** (- 31) * PI
        ephemeris.crc           = twosComp2dec(subframeBits[180:196]) * 2 ** (- 5)
        ephemeris.omega         = twosComp2dec(subframeBits[196:204] + subframeBits[210:234]) * 2 ** (- 31) * PI
        ephemeris.omegaDot      = twosComp2dec(subframeBits[240:264]) * 2 ** (- 43) * PI
        ephemeris.iDot          = twosComp2dec(subframeBits[278:292]) * 2 ** (- 43) * PI
        ephemeris.subframe3Flag = True

    elif subframeID == 4:
        # Almanac, ionospheric model, UTC parameters.
        # SV health (PRN: 25-32).
        # Not decoded at the moment.
        # TODO
        pass
    elif subframeID == 5:
        # SV almanac and health (PRN: 1-24).
        # Almanac reference week number and time.
        # Not decoded at the moment.
        # TODO
        pass
    else: 
        print(f"Unrecognised suframe ID {subframeID}.")

    # Actualize TOW
    # Compute the time of week (TOW) of the first sub-frames in the array
    # - The transmitted TOW is actual TOW of the next subframe and we need the TOW of the first subframe in this data block
    # - The TOW written in the message is referred to very begining of the subframe, meaning the first bit of the preambule.
    # -> So we remove 6 seconds to have the TOW of the current subframe
    # - Also need to realign the TOW to the current process time 
    # -> account for all the bits since last subframe was decoded
    tow  = bin2dec(subframeBits[30:47]) * 6 
    tow -= 6
    #ephemeris.tow = tow
    
    #print(f"Subframe {subframeID} decoded for satellite G{self.svid} (TOW = {eph.tow}).") 
    #logging.getLogger(__name__).info(f"Subframe {subframeID} decoded for satellite G{self.svid} (TOW = {eph.tow}).")

    return tow, ephemeris

# =====================================================================================================================


