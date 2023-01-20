
#include <stdio.h>
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

void generateReplica(double* time, int size_time, int samplesRequired, float carrierFrequency, 
                                float remCarrierPhase, complex double* replica){
    
    float tmp = 0.0;
    for(int i=0; i < size_time; i++){
        tmp = -(carrierFrequency * 2.0 * PI * time[i]) + remCarrierPhase;
        //replica[i] = cexp(tmp * I);
    }
    return;
}

// --------------------------------------------------------------------------------------------------------------------



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


