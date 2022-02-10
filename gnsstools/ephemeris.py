from cmath import isnan, nan
import numpy as np

class Ephemeris:

    PREAMBULE_BITS = np.array([1, -1, -1, -1, 1, -1, 1, 1])
    MS_IN_NAV_BIT = 20 # TODO move to configuration file
    MS_IN_SUBFRAME = 6000

    # -------------------------------------------------------------------------

    def __init__(self, track):

        self.track = track
        self.prn = track.prn
        self.firstSubFrame = np.NaN
        self.isActive = False

        return

    # -------------------------------------------------------------------------

    def fromRawNavigationMessage(self):

        # Find the preambule in the signal
        self.findPreambles(self.track.iPrompt)
        
        return

    # -------------------------------------------------------------------------

    def findPreambles(self, signal):
        preambule = np.kron(self.PREAMBULE_BITS, np.ones(20))

        iPromptBits = np.zeros_like(signal)
        iPromptBits[signal >  0] =  1
        iPromptBits[signal <= 0] = -1

        correlation = np.correlate(iPromptBits, np.pad(preambule, (0, iPromptBits.size - preambule.size), 'constant'), mode='full')

        # Take only what we want, the first part is due to 0 padding
        # Every correlation results below 153 is probably not the preambule
        # Maybe should use the max value instead? 153 feels very hardcoded
        size = int((len(correlation) + 1) / 2)
        index = (np.abs(correlation[size-1:size*2]) > 153).nonzero()[0] 

        # Analyse the premabules 
        for i in index:
            # Pre-check based on sub-frame length
            if not (index - i == self.MS_IN_SUBFRAME).any():
                continue
            # Since each bit is 20 ms, there 20 values available for each bit.
            # To help decoding in noisy signal, all these bits are accumulated.
            # 62 bits taken in total for parity check
            # Take the last 2 bits in previous subframe
            idx_start = i - (2*self.MS_IN_NAV_BIT)
            # Take the first two 30 bit words (TLM + HOW)
            idx_stop  = i + (30 * 2 * self.MS_IN_NAV_BIT) 

            bits = signal[idx_start:idx_stop].copy()

            # Reshape in 20 arrays of 62 bits
            bits = bits.reshape(20, -1, order='F')

            # Accumulate the results
            bits = bits.sum(0)

            bits[bits >  0] =  1
            bits[bits <= 0] = -1

            # TODO Weird fix, should be removed
            bits = -bits 

            if self.navPartyChk(bits[:32]) != 0 and self.navPartyChk(bits[30:62]) !=0:
                self.firstSubFrame = i
                self.isActive = True
                break
        
        if isnan(self.firstSubFrame):
            print(f"Could not find preambule in satellite tracked {self.prn}.")

        return 


    # -------------------------------------------------------------------------
    @staticmethod
    def navPartyChk(ndat):
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