
import numpy as np

# =====================================================================================================================
# FLL LOCK INDICATOR

def FLL_Lock_Borre(iprompt, iprompt_prev, qprompt, qprompt_prev, fll_lock_prev, alpha=0.01):
    
    # Compute FLL lock
    fll_lock  = iprompt * iprompt_prev - qprompt * qprompt_prev; 
    fll_lock *= np.sign(iprompt * iprompt_prev + qprompt * qprompt_prev)
    fll_lock /= (iprompt**2 + qprompt**2)
    fll_lock  = abs(fll_lock)

    # Pass through low-pass filter
    fll_lock = (1 - alpha) * fll_lock_prev + alpha * fll_lock

    return fll_lock

# =====================================================================================================================
# PLL LOCK INDICATOR

def PLL_Lock_Borre(iprompt, qprompt, pll_lock_prev, alpha=0.01):

    # Narrow Band Difference
    nbd = iprompt**2 - qprompt**2

    # Narrow Band Power
    nbp = iprompt**2 + qprompt**2
    
    pll_lock = nbd / nbp

    # Pass through low-pass filter
    pll_lock = (1 - alpha) * pll_lock_prev + alpha * pll_lock

    return pll_lock

# =====================================================================================================================
# CN0 

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

def CN0_Beaulieu(ratio:float, N:int, T:float, old:float):
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

    cn0 = lowPassFilter(cn0, old, alpha=0.1)

    return cn0

# =====================================================================================================================
# LOW PASS FILTER

def lowPassFilter(new:float, old:float, alpha:float):
    """
    Implementation of low pass filter. 

    Args:
        new (float): New value.
        old (float): Old accumulated value.
        alpha (float): Forgetting rate.
    
    Returns:
        filtered (float): Filtered value.

    Raises:
        None
    """

    filtered = (1 - alpha) * old + alpha * new

    return filtered

# =====================================================================================================================