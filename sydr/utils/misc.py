
import numpy as np

# =====================================================================================================================

def bin2dec(binaryStr):
    assert isinstance(binaryStr, str)
    return int(binaryStr, 2)

# =====================================================================================================================

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

# =====================================================================================================================

# preallocate empty array and assign slice by chrisaycock
def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = arr[-num:]
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = arr[-num:]
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

# =====================================================================================================================

# Quantize a signal to the required number of bits
def quantize(signal, n_bits):

    scale_factor = (2**n_bits // 2) / np.max(np.abs(signal)) 
    signal_quantized = np.round(signal * scale_factor - 0.5).astype(int)

    return signal_quantized, scale_factor

# =====================================================================================================================

def multiply_axc(array_1, array_2, axc_mult):

    result = np.zeros_like(array_1)
    for i in range(len(array_1)):
        result[i] = axc_mult(array_1[i], array_2[i])

    return 