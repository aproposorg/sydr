
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

// GPS' definition of Pi
#define PI 3.1415926535898   

void readValues(complex double * rfdata, int rfsize){
    // Read values
    for(int i=0; i < 8; i++){
        printf("%f + %fi \n", creal(rfdata[i]), cimag(rfdata[i]));
    }
} 

// --------------------------------------------------------------------------------------------------------------------

/*
@brief
@param 
@return void.
*/
void generateReplica(double* time,
                     size_t samplesRequired,
                     double carrierFrequency,
                     double remCarrierPhase,
                     double* r_remCarrierPhase,
                     complex double* r_replica) 
{
    double temp;

    for (size_t i=0; i < samplesRequired; i++)
    {
        temp = -(carrierFrequency * 2.0 * PI * time[i]) + remCarrierPhase;
        r_replica[i] = cexp(I * temp);
    }
    temp = -(carrierFrequency * 2.0 * PI * time[samplesRequired]) + remCarrierPhase;

    *r_remCarrierPhase = fmod(temp, 2*PI);

    return;
}

// --------------------------------------------------------------------------------------------------------------------

/*
@brief Perform a correlation operation between the code and I/Q signal.
@param iSignal           Array with the real part of the signal.
@param qSignal           Array with the imaginary part of the signal.
@param code              Array with the PRN code.
@param size              Size of the arrays (iSignal, qSignal, code, iCorr, qCorr).
@param codeStep          The step size between the code frequency and the sampling frequency.
@param remCodePhase      Remaining code phase from last loop.
@param correlatorSpacing Correlator spacing in code chip.
@param r_iCorr             Array with correlation result for the real part.
@param r_qCorr             Array with correlation result for the imaginary part.
@return void.
*/
void getCorrelator(double* iSignal,
                   double* qSignal,
                   int* code,
                   size_t size, // samplesRequired
                   double codeStep,
                   double remCodePhase, 
                   double correlatorSpacing,
                   double* r_iCorr,
                   double* r_qCorr)
{
    size_t codeIdx = 0;
    
    double start = remCodePhase + correlatorSpacing;
    double stop = size * codeStep + remCodePhase + correlatorSpacing;
    double step = (stop - start) / size;

    for (size_t idx=0; idx < size; idx++)
    {
        codeIdx = ceil(start + step * idx);

        r_iCorr[idx] = code[codeIdx] * iSignal[idx];
        r_qCorr[idx] = code[codeIdx] * qSignal[idx];
    }

    return;
}

// --------------------------------------------------------------------------------------------------------------------

/*
@brief
@param
@return void.
*/
void generateCarrier(complex double* rfData,
                     complex double* replica,
                     size_t size,
                     double* r_iSignal,
                     double* r_qSignal)
{
    complex double temp;

    for (size_t i=0; i < size; i++)
    {
        temp = rfData[i] * replica[i];

        r_iSignal[i] = creal(temp);
        r_qSignal[i] = cimag(temp);
    }

    return;
}

// --------------------------------------------------------------------------------------------------------------------

/*
@brief
@param
@return void.
*/
void delayLockLoop(double iEarly,
                   double qEarly,
                   double iLate,
                   double qLate,
                   double dllTau1,
                   double dllTau2,
                   double pdiCode,
                   double codeNCO,
                   double codeError,
                   double codeFrequency,
                   double* r_codeNCO,
                   double* r_codeError,
                   double* r_codeFrequency)
{
    double newCodeError = (sqrt(pow(iEarly, 2) + pow(qEarly, 2)) -
                           sqrt(pow(iLate,  2) + pow(qLate,  2))) /
                          (sqrt(pow(iEarly, 2) + pow(qEarly, 2)) +
                           sqrt(pow(iLate,  2) + pow(qLate,  2)));

    *r_codeNCO  = codeNCO;
    *r_codeNCO += dllTau2 / dllTau1 * (newCodeError - codeError);
    *r_codeNCO += pdiCode / dllTau1 *  newCodeError;

    *r_codeError = newCodeError;

    *r_codeFrequency = codeFrequency - *r_codeNCO;

    return;
}

// --------------------------------------------------------------------------------------------------------------------

/*
@brief
@param
@return void.
*/
void phaseLockLoop(double iPrompt,
                   double qPrompt,
                   double pllTau1,
                   double pllTau2,
                   double pdiCarrier,
                   double carrierNCO,
                   double carrierError,
                   double initialFrequency,
                   double* r_carrierNCO,
                   double* r_carrierError,
                   double* r_carrierFrequency)
{
    double newCarrierError = atan(qPrompt / iPrompt) / 2.0 / PI;

    *r_carrierNCO  = carrierNCO;
    *r_carrierNCO += pllTau2 / pllTau1 * (newCarrierError - carrierError);
    *r_carrierNCO += pdiCarrier / pllTau1 * newCarrierError;

    *r_carrierError = carrierError;

    *r_carrierFrequency = initialFrequency + *r_carrierNCO;

    return;
}

// --------------------------------------------------------------------------------------------------------------------

/*
@brief
@param
@return void.
*/
void getLoopCoefficients(double loopNoiseBandwidth,
                         double dumpingRatio,
                         double loopGain,
                         double* r_tau1,
                         double* r_tau2)
{
    double wn = loopNoiseBandwidth * 8.0 * dumpingRatio / (4.0 * pow(dumpingRatio, 2) + 1);

    *r_tau1 = loopGain / pow(wn, 2);
    *r_tau2 = 2.0 * dumpingRatio / wn;

    return;
}

/// ===================================================================================================================

int main(){
    printf("Hello World!\n");

    int rfsize = 8;
    complex double rfdata[rfsize];

    // Put values
    for(int i=0; i < rfsize; i++){
        rfdata[i] = i + i*2.0 * I;
    }

    readValues(rfdata, rfsize);

    return 0;
}
