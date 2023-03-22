
import numpy as np

# =====================================================================================================================

def CN0_NWPR(iPrompt:float, qPrompt:float, nbPrompt):
    """
    C/N0 estimator based on the Narrow-band Wide-band Power Ratio (NWPR) algorithm. 
    See reference [Borre, 2023] for definition.
    """

    # Compute Narrow-Band Power (NBP) and Wide-Band Power (WBP)
    # TODO

    return

# =====================================================================================================================

def NWPR(iPrompt:np.array, qPrompt:np.array):
    """
    Narrow-band Wide-band Power Ratio (NWPR) estimation. I/Q prompt are assumed free of bit transition.
    See reference [Borre, 2023] for definition.

    Args:
        iPrompt (np.array): In-Phase Prompt correlator results.
        qPrompt (np.array): Quadraphase Prompt corrector results.

    Return: 
        normalisedPower (float): Normalised Power Ratio.

    Raises:
        None
    """

    # Narrow-Band Power (NBP) 
    nbp = np.sum(iPrompt)**2 + np.sum(qPrompt)**2

    # Wide-Band Power (WBP)
    wbp = np.sum(np.square(iPrompt)) + np.sum(np.square(qPrompt))
    
    # Normalised Power (NP)
    normalisedPower = nbp / wbp

    return normalisedPower

# =====================================================================================================================