

def bin2dec(binaryStr):
    assert isinstance(binaryStr, str)
    return int(binaryStr, 2)

# -----------------------------------------------------------------------------

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