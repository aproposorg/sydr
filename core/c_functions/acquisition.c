
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

// --------------------------------------------------------------------------------------------------------------------

/*
@brief
@param
@return void.
*/
void setSatellite(double* code,
                  size_t size,
                  complex double* codeFFT)
{
    return;
}

// --------------------------------------------------------------------------------------------------------------------

/*
@brief
@param
@return void.
*/
void PCPS()
{
    return;
}

// --------------------------------------------------------------------------------------------------------------------

/*
@brief
@param
@return void.
*/
void twoCorrelationPeakComparison(double* correlationMap, // 2D (s_frequencyBins, s_correlationMap)
                                  size_t s_correlationMap,
                                  double* frequencyBins,
                                  size_t s_frequencyBins,
                                  long long samplesPerCode,
                                  long long samplesPerCodeChip,
                                  double interFrequency,
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
