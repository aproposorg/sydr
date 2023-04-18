
import numpy as np

# =====================================================================================================================

def generateReplica(time:np.array, nbSamples:int, carrierFrequency:float, remCarrier:float, ):

    # Generate replica and mix signal
    time = time[0:nbSamples+1]
    temp = -(carrierFrequency * 2.0 * np.pi * time) + remCarrier

    remCarrier = temp[nbSamples] % (2 * np.pi)
    replica = np.exp(1j * temp[:nbSamples])

    return replica, remCarrier

# =====================================================================================================================

def getCorrelator(iSignal:np.array, qSignal:np.array, correlatorSpacing:float, code:np.array, remainingCode:float,\
                  codeStep:float, nbSamples:int):
    """
    Return the I and Q correlation of an RF signal with a sampled code.
    """

    idx = np.ceil(
            np.linspace(remainingCode + correlatorSpacing, nbSamples * codeStep + remainingCode + correlatorSpacing, \
            nbSamples, endpoint=False)).astype(int)
    tmpCode = code[idx]

    iCorr  = np.sum(tmpCode * iSignal)
    qCorr  = np.sum(tmpCode * qSignal)

    return iCorr, qCorr

# =====================================================================================================================

def LoopFiltersCoefficients(loopNoiseBandwidth:float, dampingRatio:float, loopGain:float):
    """
    Return the loop filters coefficients. See reference [Borre, 2007].

    Args:
        loopNoiseBandwidth (float): Loop Noise Bandwith parameter
        dampingRatio (float): Damping Ratio parameter, a.k.a. zeta
        loopGain (float): Loop Gain parameter
    
    Returns
        tau1 (float): Loop filter coefficient (1st)
        tau2 (float): Loop filter coefficient (2nd)

    Raises:
        None
    """

    Wn = loopNoiseBandwidth * 8.0 * dampingRatio / (4.0 * dampingRatio**2 +1)
    
    tau1 = loopGain / Wn**2
    tau2 = 2.0 * dampingRatio / Wn

    return tau1, tau2

# =====================================================================================================================

def EPL_nonvector(rfData:np.array, code:np.array, samplingFrequency:float, carrierFrequency:float, remainingCarrier:float, \
        remainingCode:float, codeStep:float, correlatorsSpacing:tuple):
    
    rfData = np.squeeze(rfData)
    
    nbSamples = len(rfData)
    correlatorResults = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for idx in range(nbSamples):
        # Generate replica
        temp = -(carrierFrequency * 2.0 * np.pi * (idx/samplingFrequency)) + remainingCarrier
        replica = np.exp(1j * temp)

        # Mix replica and RF signal
        signal = replica * rfData[idx]
        iSignal = np.real(signal)
        qSignal = np.imag(signal)

        # Perform correlation
        for i in range(len(correlatorsSpacing)):
            codeIdx = int(np.ceil(remainingCode + correlatorsSpacing[i] + idx*codeStep))
            correlatorResults[i*2]   += code[codeIdx] * iSignal
            correlatorResults[i*2+1] += code[codeIdx] * qSignal
    
    return correlatorResults

# =====================================================================================================================

def EPL(rfData:np.array, code:np.array, samplingFrequency:float, carrierFrequency:float, remainingCarrier:float, \
        remainingCode:float, codeStep:float, correlatorsSpacing:tuple):
    
    rfData = np.squeeze(rfData)
    
    nbSamples = len(rfData)
    correlatorResults = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Generate replica
    time = np.arange(0.0, nbSamples) / samplingFrequency
    replica = np.exp(1j * (-(carrierFrequency * 2.0 * np.pi * time) + remainingCarrier))

    # Mix replica
    signal = replica * rfData
    iSignal = np.real(signal)
    qSignal = np.imag(signal)

    # Perform correlation
    for i in range(len(correlatorsSpacing)):
        shift = remainingCode + correlatorsSpacing[i]
        codeIdx = np.ceil(np.linspace(shift, codeStep * nbSamples + shift, nbSamples, endpoint=False)).astype(int)
        correlatorResults[i*2]   = np.sum(code[codeIdx] * iSignal)
        correlatorResults[i*2+1] = np.sum(code[codeIdx] * qSignal)
    
    return correlatorResults

# =====================================================================================================================

def DLL_NNEML(iEarly:float, qEarly:float, iLate:float, qLate:float):
    """
    Delay Lock Loop implementation, using a Normalize Noncoherent Early Minus Late (NNEML) discriminator.
    See reference [Borre, 2023], p.65
    """

    codeError = (np.sqrt(iEarly**2 + qEarly**2) - np.sqrt(iLate**2 + qLate**2)) / \
                (np.sqrt(iEarly**2 + qEarly**2) + np.sqrt(iLate**2 + qLate**2))

    return codeError

# =====================================================================================================================

def PLL_costa(iPrompt:float, qPrompt:float):
    """
    Phase Lock Loop implementation, using a Costas discriminator. 
    See reference [Borre, 2023], p.59
    """

    phaseError = np.arctan(qPrompt / iPrompt) / 2.0 / np.pi

    return phaseError

# =====================================================================================================================

def FLL_ATAN2(iPrompt:float, qPrompt:float, iPromptPrev:float, qPromptPrev:float, deltaT:float):
    
    frequencyError = np.arctan2(iPromptPrev * qPrompt - qPromptPrev * iPrompt,
                                iPromptPrev * iPrompt + qPromptPrev * qPrompt) / 2.0 / np.pi
    frequencyError /= deltaT 

    return frequencyError

# =====================================================================================================================

def BorreLoopFilter(input:float, memory:float, tau1:float, tau2:float, pdi:float):

    # Update NCO frequency
    output  = tau2 / tau1 * (input - memory)
    output += pdi / tau1 * input

    return output

# =====================================================================================================================

def firstOrderDLF(input, w0):
    """
    Perform a first order Digital Loop Filter (DLF).
    See [Kaplan, 2006], p181.
    """

    output = input * w0

    return output

# =====================================================================================================================

def secondOrferDLF(input, w0, a2, integrationTime, memory):
    """
    """

    c1 = w0**2
    c2 = a2 * w0

    # First branch
    _memoryUpdate = input * c1 * integrationTime
    output = (_memoryUpdate + memory) / 2
    memory = _memoryUpdate # TODO is it really an addition or we replace the previous memory?

    # Second branch
    output += input * c2

    return output, memory

# =====================================================================================================================

def thirdOrderDLF(input:float, w0:float, a3:float, b3:float, integrationTime:float, memory1:float, memory2:float):
    """
    """

    c1 = w0**3
    c2 = a3 * w0**2
    c3 = b3 * w0

    # First branch
    _memoryUpdate = input * c1 * integrationTime
    output = (_memoryUpdate + memory1) / 2
    memory1 = _memoryUpdate

    # Second branch
    _memoryUpdate = (output + input * c2) * integrationTime
    output = (_memoryUpdate + memory2) / 2
    memory2 = _memoryUpdate

    # Third branch
    output += input * c3

    return output, memory1, memory2

# =====================================================================================================================

def FLLassistedPLL_2ndOrder(phaseInput:float, freqInput:float, w0f:float, w0p:float, a2:float, integrationTime:float,
                            velMemory:float):
    """
    Perform a Digital Loop Filter (DLF) for frequency tracking, using a 2nd order PLL assisted by 1st order FLL.
    See [Kaplan, 2006], p180-182.

    Args:
        phaseInput (float): Phase error input
        freqInput (float): Frequency error input
        w0f (float): Loop filter natural radian frequency for FLL
        w0p (float): Loop filter natural radian frequency for PLL
        a2 (float): 2nd order DLF constant
        integrationTime (float): Coherent integration time 
        velMemory (float): Velocity accumulator memory

    Returns:
        output (float): Result from the loop filter
        velMemory (float): Updated velocity accumulator memory

    Raises:
        None

    """

    # 2nd order PLL, 1st order FLL
    # First branch
    _memoryUpdate = (phaseInput * w0p**2 + freqInput * w0f) * integrationTime
    output = (_memoryUpdate + velMemory) / 2
    velMemory = _memoryUpdate
    
    # Second branch
    output += phaseInput * a2 * w0p

    return output, velMemory

# =====================================================================================================================

def FLLassistedPLL_3rdOrder(phaseInput:float, freqInput:float, w0f:float, w0p:float, a2:float, a3:float, b3:float, 
                            integrationTime:float, velMemory:float, accMemory:float):
    """
    Perform a Digital Loop Filter (DLF) for frequency tracking, using a 3rd order PLL assisted by 2nd order FLL.
    See [Kaplan, 2006], p180-182.

    Args:
        phaseInput (float): Phase error input
        freqInput (float): Frequency error input
        w0f (float): Loop filter natural radian frequency for FLL
        w0p (float): Loop filter natural radian frequency for PLL
        a2 (float): 2nd order DLF constant
        a3 (float): 3rd order DLF constant
        b3 (float): 3rd order DLF constant
        integrationTime (float): Coherent integration time 
        velMemory (float): Velocity accumulator memory
        accMemory (float): Acceleration accumulator memory

    Returns:
        output (float): Result from the loop filter
        velMemory (float): Updated velocity accumulator memory
        accMemory (float): Updated acceleration accumulator memory

    Raises:
        None

    """

    # 3rd order PLL, 2nd order FLL
    # First branch
    _memoryUpdate = (phaseInput * w0p**3 + freqInput * w0f**2) * integrationTime
    output = (_memoryUpdate + accMemory) / 2
    accMemory = _memoryUpdate
    
    # Second branch
    _memoryUpdate = (output + (phaseInput * a3 * w0p**2 + freqInput * a2 * w0f)) * integrationTime
    output = (_memoryUpdate + velMemory) / 2
    velMemory = _memoryUpdate

    # Third branch
    output += phaseInput * b3 * w0p

    return output, velMemory, accMemory

# =====================================================================================================================



# =====================================================================================================================