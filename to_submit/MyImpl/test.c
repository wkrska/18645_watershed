// #include <opencv2/core.hpp> // to be able to use Mat class
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "immintrin.h"

#include "morph_kernel.c"
#include "kernel_alts.c"

#define BATCH (4) // the number of SIMD reads per iteration
#define D_WIDTH (8)
#define SIMD_N_ELEM (256/D_WIDTH) // the number of addresses to step for each SIMD read

#ifndef ROWS
#define ROWS 32
#endif
#ifndef COLS 
#define COLS 5*SIMD_N_ELEM
#endif

#ifndef RUNS
#define RUNS 1//0000000
#endif


#define MAX_FREQ 3.4
#define BASE_FREQ 2.4

#define RUN_ROW
// #define RUN_PACK
#define RUN_COL
// #define RUN_UNPACK

#define DEBUG 1
// #define PRINTMAT

//timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

bool check(int8_t* a, int8_t* b) {
    bool correct = true;
    for (int i = 0; i < ROWS*COLS; i++)
        correct &= (a[i]==b[i]);
    return correct;
}

int main(int argc, char** argv) {
    //////////////////////////////////
    // Parallelization
    //////////////////////////////////
    // Divide optimally
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int root = 0;
    // Do all initialization on root
    if (world_rank == root) {

    }

    // make sure that minimums are met for image
    assert(COLS >= SIMD_N_ELEM);

    int8_t *in, *out, *buff;
    posix_memalign((void**) &in, 64, ROWS * COLS * sizeof(int8_t));
    posix_memalign((void**) &out, 64, ROWS  * COLS * sizeof(int8_t));
    posix_memalign((void**) &buff, 64, ROWS  * COLS * sizeof(int8_t));

    // Fill in with data
    fill: for (int i = 0; i != ROWS*COLS; ++i)  { 
        #if (DEBUG == 1)
        int tile_size = 4;
        // Checker board with tiles of above size
        in[i] = (((i/(COLS*tile_size)) % 2 == 0) ^ (((i%(COLS))/tile_size) % 2 == 0)) ? 0 : ~0;
        #elif (DEBUG == 2)
        in[i] = i;
        #else
        in[i] = ((int8_t) rand())/ ((int8_t) RAND_MAX);
        #endif

        out[i] = 0;
        buff[i] = 0;
    }

    unsigned long long t0, t1;
    unsigned long long timer;

    
    #ifdef PRINTMAT
    printf("Unpacked Input:\n");
    mat_print(COLS, ROWS, in);
    printf("\n\n");
    #endif

    #ifdef RUN_ROW
        // Benchmark Horizontal Kernel
        {
            timer = ~0;
            rows: for (int i = 0; i < RUNS; i++) {
                t0 = rdtsc();
                rows_kernel(COLS,ROWS,in,out);
                t1 = rdtsc();

                timer = ((t1-t0) < timer) ? t1-t0 : timer;
            }
            printf("Horizontal Efficiency: %f\n", (double) (BASE_FREQ*3/2*ROWS*COLS/SIMD_N_ELEM)/(MAX_FREQ*timer));
        }
        #ifdef PRINTMAT
            printf("Hor. Output:\n");
            mat_print(COLS, ROWS, out);
            printf("\n\n");
        #endif
    #else
    for (int i = 0; i < ROWS*COLS; i++)
        out[i] = in[i];
    #endif

    

    #ifdef RUN_PACK
        // Benchmark packing
        timer = ~0;
        pack: for (int i = 0; i < RUNS; i++) {
            t0 = rdtsc();
            pack(COLS,ROWS,out,buff);
            t1 = rdtsc();

            timer = ((t1-t0) < timer) ? t1-t0 : timer;
        }
        printf("Packing Efficiency: %f\n", (double) (BASE_FREQ*ROWS*COLS/SIMD_N_ELEM)/(MAX_FREQ*timer));
        #ifdef PRINTMAT
            printf("Packed Input:\n");
            mat_print(ROWS*SIMD_N_ELEM,COLS/SIMD_N_ELEM, buff);
            printf("\n\n");
        #endif
    #else
    for (int i = 0; i < ROWS*COLS; i++)
        buff[i] = out[i];
    #endif
    

    #ifdef RUN_COL
        // Benchmark Vertical Kernel
        timer = ~0;
        cols: for (int i = 0; i < RUNS; i++) {
            t0 = rdtsc();
            cols_kernel(COLS,ROWS,buff,buff);
            t1 = rdtsc();

            timer = ((t1-t0) < timer) ? t1-t0 : timer;
        }
        printf("Vertical Efficiency: %f\n", (double) (BASE_FREQ*ROWS*COLS/SIMD_N_ELEM)/(MAX_FREQ*timer));
        #ifdef PRINTMAT
            printf("Packed Output:\n");
            mat_print(ROWS*SIMD_N_ELEM,COLS/SIMD_N_ELEM,buff);
            printf("\n\n");
        #endif
    #endif

    
    #ifdef RUN_UNPACK
        // Benchmark unpacking
        timer = ~0;
        upack: for (int i = 0; i < RUNS; i++) {
            t0 = rdtsc();
            unpack(COLS, ROWS, buff, out);
            t1 = rdtsc();

            timer = ((t1-t0) < timer) ? t1-t0 : timer;
        }
        printf("Unpacking Efficiency: %f\n", (double) (BASE_FREQ*ROWS*COLS/SIMD_N_ELEM)/(MAX_FREQ*timer));
        #ifdef PRINTMAT
            printf("Unpacked Output:\n");
            mat_print(COLS,ROWS,out);
            printf("\n\n");
        #endif
    #else
    for (int i = 0; i < ROWS*COLS; i++)
        out[i] = buff[i];
    #endif

} 