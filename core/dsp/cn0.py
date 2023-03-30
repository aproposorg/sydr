
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

def NWPR(iPromptSum:float, qPromptSum:float, iPromptSum2:float, qPromptSum2:float):
    """
    Narrow-band Wide-band Power Ratio (NWPR) estimation. I/Q prompt are assumed free of bit transition.
    See reference [Borre, 2023] for definition.

    Args:
        iPromptSum (float): Sum of In-Phase Prompt correlator results.
        qPromptSum (float): Sum of Quadraphase Prompt correlator results.
        iPromptSum2 (float): Sum of squares of In-Phase Prompt correlator results.
        qPromptSum2 (float): Sum of squares of Quadraphase Prompt correlator results.

    Return: 
        normalisedPower (float): Normalised Power Ratio.

    Raises:
        None
    """

    # Narrow-Band Power (NBP) 
    nbp = iPromptSum**2 + qPromptSum**2

    # Wide-Band Power (WBP)
    wbp = iPromptSum2 + qPromptSum2
    
    # Normalised Power (NP)
    normalisedPower = nbp / wbp

    return normalisedPower

# =====================================================================================================================