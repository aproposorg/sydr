
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include <assert.h>

// GPS' definition of Pi
#define PI 3.1415926535898

void readValues(complex double * rfdata, int rfsize){
    // Read values
    for(int i=0; i < 8; i++){
        printf("%f + %fi \n", creal(rfdata[i]), cimag(rfdata[i]));
    }
} 

// --------------------------------------------------------------------------------------------------------------------
// TODO Coud be simplified by just recomputing the value of time in for loop instead of giving an array.
/*
@brief
@param time              Array of length `size+1` with time values ranging from 0 to (size+1) / sampling frequency 
@param size              Length of the arrays 
@param carrierFrequency  Carrier frequency of the signal (i.e. remaining due to Doppler effect)
@param remCarrierPhase   Remaining carrier phase from last loop (initial value is 0.0)
@param r_remCarrierPhase Return of the remCarrierPhase
@param r_replica         Return of the replica array of length `size`
@return void.
*/
void generateReplica(double* time,
                     size_t size,
                     double carrierFrequency,
                     double remCarrierPhase,
                     double* r_remCarrierPhase,
                     complex double* r_replica) 
{
    double temp;

    for (size_t i=0; i < size; i++)
    {
        temp = -(carrierFrequency * 2.0 * PI * time[i]) + remCarrierPhase;
        r_replica[i] = cexp(I * temp);
        //printf("%f\n", temp);
        //printf("%f + i%f\n", creal(r_replica[i]), cimag(r_replica[i]));
    }
    temp = -(carrierFrequency * 2.0 * PI * time[size]) + remCarrierPhase; //0.0094247779607694003

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

// --------------------------------------------------------------------------------------------------------------------

bool test_generateReplica(){
    /*
    Unit test function of generateReplica.
    */
    int size = 5;
    int samplingFrequency = 1e7;
    double carrierFrequency = -1500.0;
    double remCarrierPhase = 0.0;
    double time[size+1];
    double r_remCarrierPhase = 0.0;
    complex double r_replica[size];
    complex double r_replica_truth[size];
    float epsilon = 1e-8;
    bool success = true;
    
    // init 
    for(int i=0; i < size+1; i++){
        time[i] = (double) i / samplingFrequency;
    }
    for(int i=0; i < size; i++){
        r_replica[i] = 0.0;
    }

    // test function
    generateReplica(time, size, carrierFrequency, remCarrierPhase, &r_remCarrierPhase, r_replica);

    // Assert
    r_replica_truth[0] = 1 + I * 0.0;
    r_replica_truth[1] = 0.9999995558678348 + I * 0.000942477656548699;
    r_replica_truth[2] = 0.9999982234717338 + I * 0.0018849544759281136;
    r_replica_truth[3] = 0.9999960028128805 + I * 0.002827429620969703;
    r_replica_truth[4] = 0.9999928938932473 + I * 0.0037699022545064132;

    for(int i=0; i < size; i++){
        if(fabs(r_replica_truth[i] - r_replica[i]) > epsilon){
            success = false;
            break;
        }
    }

    assert(success == true);

    return success;
}

/// ===================================================================================================================

int main(){
    printf("Hello World!\n");
    
    test_generateReplica();

    // int size = 10000;
    // int samplingFrequency = 1e7;
    // double carrierFrequency = -1500.0;
    // double remCarrierPhase = 0.0;
    // double time[size+1];
    // double r_remCarrierPhase = 0.0;
    // complex double r_replica[size];
    
    // // init 
    // for(int i=0; i < size+1; i++){
    //     time[i] = (double) i / samplingFrequency;
    // }
    // for(int i=0; i < size; i++){
    //     r_replica[i] = 0.0;
    // }

    // // test function
    // generateReplica(time, size, carrierFrequency, remCarrierPhase, &r_remCarrierPhase, r_replica);

    // return 0;
}

/*
temp
0.0
0.0009424777960769379
0.0018849555921538759
0.0028274333882308137
0.0037699111843077517
0.00471238898038469
0.005654866776461627
0.006597344572538565
0.007539822368615503
0.00848230016469244
0.00942477796076938

remCarrierPhase
0.00942477796076938

replica
(1+0j)
(0.9999995558678348+0.000942477656548699j)
(0.9999982234717338+0.0018849544759281136j)
(0.9999960028128805+0.002827429620969703j)
(0.9999928938932473+0.0037699022545064132j)
(0.999988896715596+0.004712371539373423j)
(0.9999840112834769+0.005654836638408881j)
(0.9999782376012297+0.00659729671445466j)
(0.999971575673983+0.007539750930357091j)
(0.9999640255076541+0.00848219844896771j)
(0.9999555871089498+0.009424638433144006j)
*/