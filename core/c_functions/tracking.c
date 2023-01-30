
#include <stdio.h>
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

void generateReplica(double* time, int size_time, 
                    int samplesRequired, 
                    float carrierFrequency, 
                    float remCarrierPhase, complex double* replica){
    
    float tmp = 0.0;
    for(int i=0; i < size_time; i++){
        tmp = -(carrierFrequency * 2.0 * PI * time[i]) + remCarrierPhase;
        replica[i] = cexp(tmp * I);
    }

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
@param iCorr             Array with correlation result for the real part.
@param qCorr             Array with correlation result for the imaginary part.
@return void.
*/
void getCorrelator(float* iSignal,
                   float* qSignal,
                   int* code,
                   int size, // samplesRequired
                   float codeStep,
                   float remCodePhase, 
                   float correlatorSpacing,
                   float* iCorr,
                   float* qCorr){

    int codeIdx = 0;
    
    float start = remCodePhase + correlatorSpacing;
    float stop = size * codeStep + remCodePhase + correlatorSpacing;
    float step = (stop - start) / size;

    for(int idx=0; idx < size; idx++){
        codeIdx = ceil(start + step * idx);

        iCorr[idx] = code[codeIdx] * iSignal[idx];
        qCorr[idx] = code[codeIdx] * qSignal[idx];
    }

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


