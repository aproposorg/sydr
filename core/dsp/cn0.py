
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

def CN0_Beaulieu(ratio:float, N:int, T:float, cn0_prev=0.0, alpha=0.2):
    """
    C/N0 estimation based on Baulieu's method.
    See reference [Falletti, 2011] for definition.

    Args:
        ratio (float): Ratio of the instaneous power of the signal-noise to the total power (Pn / Pd).
        N (integer): Number of samples used for power computations
        T (float): Integration time [seconds]

    Return:
        cn0 (float): Carrier-to-Noise (C/N0) ratio.

    Raises:
        None
    """

    lambda_c = 1 / (ratio / N)
    B_eqn = 1 / T

    cn0 = lambda_c * B_eqn

    # Pass through low-pass filter
    cn0 = (1 - alpha) * cn0_prev + alpha * cn0

    return cn0


# =====================================================================================================================