
import numpy as np

# =====================================================================================================================

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

def PLL_Lock_Borre(iprompt_sum, qprompt_sum, pll_lock_prev, alpha=0.01):

    # Narrow Band Difference
    nbd = iprompt_sum**2 - qprompt_sum**2

    # Narrow Band Power
    nbp = iprompt_sum**2 + qprompt_sum**2
    
    pll_lock = nbd / nbp

    # Pass through low-pass filter
    pll_lock = (1 - alpha) * pll_lock_prev + alpha * pll_lock

    return pll_lock