// #include <opencv2/core.hpp> // to be able to use Mat class
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "immintrin.h"

#include "morph_kernel_2.h"

#define BATCH (4) // the number of SIMD reads per iteration
#define D_WIDTH (8)
#define STEP (256/D_WIDTH) // the number of addresses to step for each SIMD read

#define ROWS 8
#define COLS 2*(256/8)

#define RUNS 1
#define DEBUG 1
#define PRINTMAT 1

#define MAX_FREQ 3.4
#define BASE_FREQ 2.4

//timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

void mat_print(int cols, int rows, int8_t *mat) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%5d", mat[j+i*cols]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    int8_t *in;
    int8_t *out, *out2;

    posix_memalign((void**) &in, 64, ROWS * COLS * sizeof(int8_t));
    posix_memalign((void**) &out, 64, ROWS  * COLS * sizeof(int8_t));
    posix_memalign((void**) &out2, 64, ROWS  * COLS * sizeof(int8_t));

    // Fill in with data
    for (int i = 0; i != ROWS*COLS; ++i)  { 
        #if DEBUG
        in[i] = i;//((i/(COLS*4)) % 2 == 0) ? ~0 : 0;
        out[i] = 0;
        #else
        in[i] = ((int8_t) rand())/ ((int8_t) RAND_MAX);
        out[i] = 0;
        #endif
    }
    #if PRINTMAT
    printf("Input Mat:\n");
    mat_print(COLS, ROWS, in);
    printf("\n\n");
    #endif

    pack(COLS,ROWS,in,out);



    // unsigned long long t0, t1;
    // unsigned long long timer = ~0;
    // for (int i = 0; i < RUNS; i++) {
    //     t0 = rdtsc();
    //     cols_kernel(COLS,ROWS,out,out);
    //     t1 = rdtsc();

    //     timer = ((t1-t0) < timer) ? t1-t0 : timer;
    // }
    // printf("Efficiency: %f\n", (double) (BASE_FREQ*ROWS)/(MAX_FREQ*timer));

    #if PRINTMAT
    printf("Output Mat:\n");
    mat_print(ROWS*STEP,COLS/STEP,out);
    #endif
    
    unpack(COLS, ROWS, out, out2);

    #if PRINTMAT
    printf("Output Mat 2:\n");
    mat_print(COLS,ROWS,out2);
    #endif
    
} 