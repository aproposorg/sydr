
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "fft8g.h"

// GPS' definition of Pi
#define PI 3.1415926535898

#ifndef FFTMAX
#define FFTMAX 16384
#define FFTMAXSQRT 128
static int ip[FFTMAXSQRT + 2];   // needed for bit twiddling in FFT
static double w[FFTMAX * 5 / 4]; // needed for cos/sin weights in FFT
#endif

// --------------------------------------------------------------------------------------------------------------------

/*
@brief Pad an input signal to a power of two length and perform the FFT/IFFT on it.
@param signal Array with a real-valued signal (will be over-written with complex values).
@param size   Size of the array.
@param doInv  Whether to perform forward or inverse FFT.
@return void.
*/
void _fft(double* signal, const size_t size, const char doInv)
{
    // Compute the size to pad the signal to
    size_t paddedSize = 1;
    while (paddedSize < size)
        paddedSize <<= 1;

    // Pad the signal with zeroes
    double paddedSignal[paddedSize];
    for (size_t i = 0; i < size; i++)
        paddedSignal[i] = signal[i];
    for (size_t i = size; i < paddedSize; i++)
        paddedSignal[i] = signal[i];

    // Perform the FFT in place
    rdft(paddedSize, doInv == 0 ? 1 : -1, paddedSignal, ip, w);

    // Return the result in the same array
    for (size_t i = 0; i < size; i++)
        signal[i] = paddedSignal[i];

    return;
}

/*
@brief Pad an input signal to a power of two length and perform the real-valued FFT on it.
@param signal Array with a real-valued signal (will be over-written with complex values).
@param size   Size of the array.
@return void.
*/
void fft(double* signal, const size_t size)
{
    _fft(signal, size, 0);
    return;
}

/*
@brief Pad an input signal to a power of two length and perform the inverse FFT on it.
@param signal Array with a complex-valued signal (will be over-written with complex values).
@param size   Size of the array.
@return void.
*/
void ifft(double* signal, const size_t size)
{
    _fft(signal, size, 1);
    return;
}

// --------------------------------------------------------------------------------------------------------------------

/*
@brief
@param
@return void.
@note Output array has size/2 complex entries.
*/
void setSatellite(const double* code,
                  const size_t size,
                  complex double* codeFFT)
{
    // Perform real-valued FFT on data in place
    double signal[size];
    for (size_t i=0; i < size; i++)
        signal[i] = code[i];
    fft(signal, size);

    // Conjugate and return the result in the output array
    for (size_t i=0; i < size; i++)
        ((double*) codeFFT)[i] = (i & 0x1) ? -signal[i] : signal[i];

    return;
}

// --------------------------------------------------------------------------------------------------------------------

/*
@brief
@param
@return void.
@note Naive implementation returns the correlation map with half number of entries in the 
      second dimension than in the Python implementation due to the real-valued FFT. This 
      may affect the results.
*/
void PCPS(const complex double* rfData,
          const complex double* codeFFT, // 1D (samplesPerCode/2,)
          const long long cohIntegration,
          const long long nonCohIntegration,
          const long long samplesPerCode,
          const double samplingPeriod,
          const double interFrequency,
          const double* frequencyBins,
          const size_t s_frequencyBins,
          double* r_correlationMap) // 2D (s_frequencyBins, samplesPerCode/2)
{
    double phasePoints[cohIntegration * samplesPerCode];
    double signalCarrier[cohIntegration * samplesPerCode];
    complex double iqSignal[cohIntegration * samplesPerCode];
    double nonCohSum[samplesPerCode/2];
    complex double cohSum[samplesPerCode/2];
    for (size_t i=0; i < cohIntegration * samplesPerCode; i++)
        phasePoints[i] = (i << 1) * PI * samplingPeriod;
    
    // Do correlation for each frequency bin
    double freq;
    for (size_t i=0; i < s_frequencyBins; i++)
    {
        freq = interFrequency - frequencyBins[i];

        // Generate carrier replica
        for (size_t j=0; j < cohIntegration * samplesPerCode; j++)
            signalCarrier[j] = cexp(-I * freq * phasePoints[j]);

        // Non-coherent integration
        for (size_t j=0; j < samplesPerCode/2; j++)
            nonCohSum[j] = 0;
        for (size_t nonCohIdx=0; nonCohIdx < nonCohIntegration; nonCohIdx++)
        {
            // Mix the carrier with the required part of the data
            for (size_t j=0; j < cohIntegration * samplesPerCode; j++)
                iqSignal[j] = signalCarrier[j] * rfData[j + nonCohIdx*cohIntegration*samplesPerCode];

            // Coherent integration
            for (size_t j=0; j < samplesPerCode/3; j++)
                cohSum[j] = 0;
            for (size_t cohIdx=0; cohIdx < cohIntegration; cohIdx++)
            {
                // Perform FFT (in-place)
                fft((double*)(iqSignal + cohIdx * samplesPerCode), samplesPerCode);

                // Correlate with C/A code (in-place)
                for (size_t j=0; j < samplesPerCode/2; j++)
                    ((complex double*)(iqSignal + cohIdx * samplesPerCode))[j] *= codeFFT[j];

                // Perform IFFT
                ifft((double*)(iqSignal + cohIdx * samplesPerCode), samplesPerCode);
                for (size_t j=0; j < samplesPerCode/2; j++)
                    cohSum[j] = ((complex double*)(iqSignal + cohIdx * samplesPerCode))[j];
            }
            for (size_t j=0; j < samplesPerCode/2; j++)
                nonCohSum[j] += cabs(cohSum[j]);
        }
        for (size_t j=0; j < samplesPerCode/2; j++)
            r_correlationMap[i * s_frequencyBins + j] = nonCohSum[j];
    }

    return;
}

// --------------------------------------------------------------------------------------------------------------------

/*
@brief
@param
@return void.
*/
void twoCorrelationPeakComparison(const double* correlationMap, // 2D (s_frequencyBins, s_correlationMap)
                                  const size_t s_correlationMap,
                                  const double* frequencyBins,
                                  const size_t s_frequencyBins,
                                  const long long samplesPerCode,
                                  const long long samplesPerCodeChip,
                                  const double interFrequency,
                                  double* r_acquisitionMetric,
                                  double* r_estimatedDoppler,
                                  double* r_estimatedFrequency,
                                  long long* r_estimatedCode,
                                  long long* r_idxEstimatedFrequency,
                                  long long* r_idxEstimatedCode)
{
    double peak1 = 0.0, peak2 = 0.0;
    size_t idx0, idx1;
    long long exclude0, exclude1;

    // Find first correlation peak
    for (size_t i=0; i < s_frequencyBins * s_correlationMap; i++)
    {
        if (correlationMap[i] > peak1)
        {
            peak1 = correlationMap[i];
            idx0 = i / s_correlationMap;
            idx1 = i % s_correlationMap;
        }
    }

    // Find second correlation peak
    exclude0 = idx1 - samplesPerCodeChip;
    exclude1 = idx1 + samplesPerCodeChip;
    if (exclude0 < 1)
    {
        for (size_t i=idx0 * s_correlationMap + exclude1; i < (idx0 + 1) * s_correlationMap; i++)
            if (correlationMap[i] > peak2)
                peak2 = correlationMap[i];
    }
    else if (exclude1 >= samplesPerCode)
    {
        for (size_t i=idx0 * s_correlationMap; i < idx0 * s_correlationMap + exclude0; i++)
            if (correlationMap[i] > peak2)
                peak2 = correlationMap[i];
    }
    else
    {
        for (size_t i=idx0 * s_correlationMap; i < idx0 * s_correlationMap + exclude0; i++)
            if (correlationMap[i] > peak2)
                peak2 = correlationMap[i];
        for (size_t i=idx0 * s_correlationMap + exclude1; i < (idx0 + 1) * s_correlationMap; i++)
            if (correlationMap[i] > peak2)
                peak2 = correlationMap[i];
    }

    // Compute acquisition metric
    *r_estimatedDoppler = -frequencyBins[idx0];
    *r_estimatedCode = idx1;
    *r_acquisitionMetric = peak1 / peak2;
    *r_idxEstimatedFrequency = idx0;
    *r_estimatedFrequency = interFrequency + *r_estimatedDoppler;
    *r_idxEstimatedCode = idx1;

    return;
}

/// ===================================================================================================================

int main()
{
    printf("Hello World!\n");

    return 0;
}
