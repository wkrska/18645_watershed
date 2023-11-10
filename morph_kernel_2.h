#include <immintrin.h>
#include <stdint.h>

#define BATCHES (4) // the number of SIMD reads per iteration
#define D_WIDTH (8)
#define SIMD_N_ELEM (256/D_WIDTH) // the number of addresses to SIMD_N_ELEM for each SIMD read

inline void cols_kernel
(
    int cols,
    int rows,
    int8_t* restrict in,
    int8_t* restrict out
) {
    // To hold all new values
    __m256d ld0a, ld1a, ld2a, ld3a;
    __m256d ld0b, ld1b, ld2b, ld3b;
    // To hold all store values
    __m256d str0, str1, str2, str3;
    // To store all carry values
    __m256d car0, car1, car2, car3;

    // Zero values
    ld0a = _mm256_setzero_pd();
    ld1a = _mm256_setzero_pd();
    ld2a = _mm256_setzero_pd();
    ld3a = _mm256_setzero_pd();
    ld0b = _mm256_setzero_pd();
    ld1b = _mm256_setzero_pd();
    ld2b = _mm256_setzero_pd();
    ld3b = _mm256_setzero_pd();

    str0 = _mm256_setzero_pd();
    str1 = _mm256_setzero_pd();
    str2 = _mm256_setzero_pd();
    str3 = _mm256_setzero_pd();

    car0 = _mm256_setzero_pd();
    car1 = _mm256_setzero_pd();
    car2 = _mm256_setzero_pd();
    car3 = _mm256_setzero_pd();

    // "startup" before kernel

    // Load first BATCHES rows
    ld0b = _mm256_load_pd(in + 0 * SIMD_N_ELEM);
    ld1b = _mm256_load_pd(in + 1 * SIMD_N_ELEM);
    ld2b = _mm256_load_pd(in + 2 * SIMD_N_ELEM);
    ld3b = _mm256_load_pd(in + 3 * SIMD_N_ELEM);
    
    // Perform AND for next round (car0 is 0)
    car1 = (__m256d) _mm256_and_si256((__m256i) ld1b, (__m256i) ld0b);
    car2 = (__m256d) _mm256_and_si256((__m256i) ld2b, (__m256i) ld1b);
    car3 = (__m256d) _mm256_and_si256((__m256i) ld3b, (__m256i) ld2b);

    // Rows are packed end to end, but first BATCHES rows have been pre-processed to start the column. There will be rows/(rows read per itr) iterations
    for (int r = 1; r < rows/BATCHES; r+=2) {
        // Iteration 1
        // Load values
        // Address = in + (iteration*number of reads per iteration+my_offset) * number of values per read 
        ld0a = _mm256_load_pd(in + (r * BATCHES + 0) * SIMD_N_ELEM);
        ld1a = _mm256_load_pd(in + (r * BATCHES + 1) * SIMD_N_ELEM);
        ld2a = _mm256_load_pd(in + (r * BATCHES + 2) * SIMD_N_ELEM);
        ld3a = _mm256_load_pd(in + (r * BATCHES + 3) * SIMD_N_ELEM);

        // Perform AND for store
        str0 = (__m256d) _mm256_and_si256((__m256i) ld1b, (__m256i) car0);
        str1 = (__m256d) _mm256_and_si256((__m256i) ld2b, (__m256i) car1);
        str2 = (__m256d) _mm256_and_si256((__m256i) ld3b, (__m256i) car2);
        str3 = (__m256d) _mm256_and_si256((__m256i) ld0a, (__m256i) car3);

        // Store values
        _mm256_store_pd(out + ((r-1) * BATCHES + 0) * SIMD_N_ELEM , str0);
        _mm256_store_pd(out + ((r-1) * BATCHES + 1) * SIMD_N_ELEM , str1);
        _mm256_store_pd(out + ((r-1) * BATCHES + 2) * SIMD_N_ELEM , str2);
        _mm256_store_pd(out + ((r-1) * BATCHES + 3) * SIMD_N_ELEM , str3);

        // Perform AND for next round
        car0 = (__m256d) _mm256_and_si256((__m256i) ld0a, (__m256i) ld3b);
        car1 = (__m256d) _mm256_and_si256((__m256i) ld1a, (__m256i) ld0a);
        car2 = (__m256d) _mm256_and_si256((__m256i) ld2a, (__m256i) ld1a);
        car3 = (__m256d) _mm256_and_si256((__m256i) ld3a, (__m256i) ld2a);

        // Iteration 2
        // Load values
        // Address = in + (iteration*number of reads per iteration+my_offset) * number of values per read 
        ld0b = _mm256_load_pd(in + ((r+1) * BATCHES + 0) * SIMD_N_ELEM);
        ld1b = _mm256_load_pd(in + ((r+1) * BATCHES + 1) * SIMD_N_ELEM);
        ld2b = _mm256_load_pd(in + ((r+1) * BATCHES + 2) * SIMD_N_ELEM);
        ld3b = _mm256_load_pd(in + ((r+1) * BATCHES + 3) * SIMD_N_ELEM);

        // Perform AND for store
        str0 = (__m256d) _mm256_and_si256((__m256i) ld1a, (__m256i) car0);
        str1 = (__m256d) _mm256_and_si256((__m256i) ld2a, (__m256i) car1);
        str2 = (__m256d) _mm256_and_si256((__m256i) ld3a, (__m256i) car2);
        str3 = (__m256d) _mm256_and_si256((__m256i) ld0b, (__m256i) car3);

        // Store values
        _mm256_store_pd(out + (r * BATCHES + 0) * SIMD_N_ELEM , str0);
        _mm256_store_pd(out + (r * BATCHES + 1) * SIMD_N_ELEM , str1);
        _mm256_store_pd(out + (r * BATCHES + 2) * SIMD_N_ELEM , str2);
        _mm256_store_pd(out + (r * BATCHES + 3) * SIMD_N_ELEM , str3);

        // Perform AND for next round
        car0 = (__m256d) _mm256_and_si256((__m256i) ld0b, (__m256i) ld3a);
        car1 = (__m256d) _mm256_and_si256((__m256i) ld1b, (__m256i) ld0b);
        car2 = (__m256d) _mm256_and_si256((__m256i) ld2b, (__m256i) ld1b);
        car3 = (__m256d) _mm256_and_si256((__m256i) ld3b, (__m256i) ld2b);
    }
}

inline void rows_kernel
(
    int cols,
    int rows,
    int8_t* restrict in,
    int8_t* restrict out
) {
    // TODO
}

/*
    Packs row major image, makes SIMD columns consecutive in memory
*/
inline void pack
(
    int cols,
    int rows,
    int8_t* restrict in,
    int8_t* restrict out
) {
    inline int addrConv(int rows, int cols, int addr){
        // Read addr: row*elements in full row + col*elements in SIMD read
        // Write addr: row*elements in SIMD read + col*elements in full SIMD column
        int row = addr/cols;
        int col = addr%cols;
        return row*SIMD_N_ELEM + col*rows;
    }

    volatile __m256d ymm0 = _mm256_setzero_pd();
    volatile __m256d ymm1 = _mm256_setzero_pd();
    volatile __m256d ymm2 = _mm256_setzero_pd();
    volatile __m256d ymm3 = _mm256_setzero_pd();
    volatile __m256d ymm4 = _mm256_setzero_pd();
    volatile __m256d ymm5 = _mm256_setzero_pd();
    volatile __m256d ymm6 = _mm256_setzero_pd();
    volatile __m256d ymm7 = _mm256_setzero_pd();
    volatile __m256d ymm8 = _mm256_setzero_pd();
    volatile __m256d ymm9 = _mm256_setzero_pd();
    volatile __m256d ymm10= _mm256_setzero_pd();
    volatile __m256d ymm11= _mm256_setzero_pd();
    volatile __m256d ymm12= _mm256_setzero_pd();
    volatile __m256d ymm13= _mm256_setzero_pd();
    volatile __m256d ymm14= _mm256_setzero_pd();
    volatile __m256d ymm15= _mm256_setzero_pd();

    for (int idx = 0; idx < rows*cols; idx += SIMD_N_ELEM*16) {
        ymm0 = _mm256_load_pd(in + idx + 0 *SIMD_N_ELEM);
        ymm1 = _mm256_load_pd(in + idx + 1 *SIMD_N_ELEM);
        ymm2 = _mm256_load_pd(in + idx + 2 *SIMD_N_ELEM);
        ymm3 = _mm256_load_pd(in + idx + 3 *SIMD_N_ELEM);
        ymm4 = _mm256_load_pd(in + idx + 4 *SIMD_N_ELEM);
        ymm5 = _mm256_load_pd(in + idx + 5 *SIMD_N_ELEM);
        ymm6 = _mm256_load_pd(in + idx + 6 *SIMD_N_ELEM);
        ymm7 = _mm256_load_pd(in + idx + 7 *SIMD_N_ELEM);
        ymm8 = _mm256_load_pd(in + idx + 8 *SIMD_N_ELEM);
        ymm9 = _mm256_load_pd(in + idx + 9 *SIMD_N_ELEM);
        ymm10= _mm256_load_pd(in + idx + 10*SIMD_N_ELEM);
        ymm11= _mm256_load_pd(in + idx + 11*SIMD_N_ELEM);
        ymm12= _mm256_load_pd(in + idx + 12*SIMD_N_ELEM);
        ymm13= _mm256_load_pd(in + idx + 13*SIMD_N_ELEM);
        ymm14= _mm256_load_pd(in + idx + 14*SIMD_N_ELEM);
        ymm15= _mm256_load_pd(in + idx + 15*SIMD_N_ELEM);

        _mm256_store_pd(out + addrConv(rows, cols, idx + 0 *SIMD_N_ELEM), ymm0 );
        _mm256_store_pd(out + addrConv(rows, cols, idx + 1 *SIMD_N_ELEM), ymm1 );
        _mm256_store_pd(out + addrConv(rows, cols, idx + 2 *SIMD_N_ELEM), ymm2 );
        _mm256_store_pd(out + addrConv(rows, cols, idx + 3 *SIMD_N_ELEM), ymm3 );
        _mm256_store_pd(out + addrConv(rows, cols, idx + 4 *SIMD_N_ELEM), ymm4 );
        _mm256_store_pd(out + addrConv(rows, cols, idx + 5 *SIMD_N_ELEM), ymm5 );
        _mm256_store_pd(out + addrConv(rows, cols, idx + 6 *SIMD_N_ELEM), ymm6 );
        _mm256_store_pd(out + addrConv(rows, cols, idx + 7 *SIMD_N_ELEM), ymm7 );
        _mm256_store_pd(out + addrConv(rows, cols, idx + 8 *SIMD_N_ELEM), ymm8 );
        _mm256_store_pd(out + addrConv(rows, cols, idx + 9 *SIMD_N_ELEM), ymm9 );
        _mm256_store_pd(out + addrConv(rows, cols, idx + 10*SIMD_N_ELEM), ymm10);
        _mm256_store_pd(out + addrConv(rows, cols, idx + 11*SIMD_N_ELEM), ymm11);
        _mm256_store_pd(out + addrConv(rows, cols, idx + 12*SIMD_N_ELEM), ymm12);
        _mm256_store_pd(out + addrConv(rows, cols, idx + 13*SIMD_N_ELEM), ymm13);
        _mm256_store_pd(out + addrConv(rows, cols, idx + 14*SIMD_N_ELEM), ymm14);
        _mm256_store_pd(out + addrConv(rows, cols, idx + 15*SIMD_N_ELEM), ymm15);

        printf("read: %2d\twrite:%2d\n", (idx + 0 *SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 0 *SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 1 *SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 1 *SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 2 *SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 2 *SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 3 *SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 3 *SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 4 *SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 4 *SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 5 *SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 5 *SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 6 *SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 6 *SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 7 *SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 7 *SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 8 *SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 8 *SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 9 *SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 9 *SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 10*SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 10*SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 11*SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 11*SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 12*SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 12*SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 13*SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 13*SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 14*SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 14*SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 15*SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 15*SIMD_N_ELEM)/SIMD_N_ELEM);
    }
}

inline void unpack
(
    int cols,
    int rows,
    int8_t* restrict in,
    int8_t* restrict out
) {
    inline int addrConv(int rows, int cols, int addr){
        // Read addr: row*elements in full row + col*elements in SIMD read
        // Write addr: row*elements in SIMD read + col*elements in full SIMD column
        int row = (addr/SIMD_N_ELEM) % rows; // output row: divide by SIMD, mod by num rows
        int col = addr/rows; // output column: each input col is a row, div by row size to find col num, mult by SIMD to find addr, or just divide by num rows
        return row*SIMD_N_ELEM + col*rows;
    }

    volatile __m256d ymm0 = _mm256_setzero_pd();
    volatile __m256d ymm1 = _mm256_setzero_pd();
    volatile __m256d ymm2 = _mm256_setzero_pd();
    volatile __m256d ymm3 = _mm256_setzero_pd();
    volatile __m256d ymm4 = _mm256_setzero_pd();
    volatile __m256d ymm5 = _mm256_setzero_pd();
    volatile __m256d ymm6 = _mm256_setzero_pd();
    volatile __m256d ymm7 = _mm256_setzero_pd();
    volatile __m256d ymm8 = _mm256_setzero_pd();
    volatile __m256d ymm9 = _mm256_setzero_pd();
    volatile __m256d ymm10= _mm256_setzero_pd();
    volatile __m256d ymm11= _mm256_setzero_pd();
    volatile __m256d ymm12= _mm256_setzero_pd();
    volatile __m256d ymm13= _mm256_setzero_pd();
    volatile __m256d ymm14= _mm256_setzero_pd();
    volatile __m256d ymm15= _mm256_setzero_pd();

    for (int idx = 0; idx < rows*cols; idx += SIMD_N_ELEM*16) {
        ymm0 = _mm256_load_pd(in + idx + 0 *SIMD_N_ELEM);
        ymm1 = _mm256_load_pd(in + idx + 1 *SIMD_N_ELEM);
        ymm2 = _mm256_load_pd(in + idx + 2 *SIMD_N_ELEM);
        ymm3 = _mm256_load_pd(in + idx + 3 *SIMD_N_ELEM);
        ymm4 = _mm256_load_pd(in + idx + 4 *SIMD_N_ELEM);
        ymm5 = _mm256_load_pd(in + idx + 5 *SIMD_N_ELEM);
        ymm6 = _mm256_load_pd(in + idx + 6 *SIMD_N_ELEM);
        ymm7 = _mm256_load_pd(in + idx + 7 *SIMD_N_ELEM);
        ymm8 = _mm256_load_pd(in + idx + 8 *SIMD_N_ELEM);
        ymm9 = _mm256_load_pd(in + idx + 9 *SIMD_N_ELEM);
        ymm10= _mm256_load_pd(in + idx + 10*SIMD_N_ELEM);
        ymm11= _mm256_load_pd(in + idx + 11*SIMD_N_ELEM);
        ymm12= _mm256_load_pd(in + idx + 12*SIMD_N_ELEM);
        ymm13= _mm256_load_pd(in + idx + 13*SIMD_N_ELEM);
        ymm14= _mm256_load_pd(in + idx + 14*SIMD_N_ELEM);
        ymm15= _mm256_load_pd(in + idx + 15*SIMD_N_ELEM);

        _mm256_store_pd(out + addrConv(rows, cols, idx + 0 *SIMD_N_ELEM), ymm0 );
        _mm256_store_pd(out + addrConv(rows, cols, idx + 1 *SIMD_N_ELEM), ymm1 );
        _mm256_store_pd(out + addrConv(rows, cols, idx + 2 *SIMD_N_ELEM), ymm2 );
        _mm256_store_pd(out + addrConv(rows, cols, idx + 3 *SIMD_N_ELEM), ymm3 );
        _mm256_store_pd(out + addrConv(rows, cols, idx + 4 *SIMD_N_ELEM), ymm4 );
        _mm256_store_pd(out + addrConv(rows, cols, idx + 5 *SIMD_N_ELEM), ymm5 );
        _mm256_store_pd(out + addrConv(rows, cols, idx + 6 *SIMD_N_ELEM), ymm6 );
        _mm256_store_pd(out + addrConv(rows, cols, idx + 7 *SIMD_N_ELEM), ymm7 );
        _mm256_store_pd(out + addrConv(rows, cols, idx + 8 *SIMD_N_ELEM), ymm8 );
        _mm256_store_pd(out + addrConv(rows, cols, idx + 9 *SIMD_N_ELEM), ymm9 );
        _mm256_store_pd(out + addrConv(rows, cols, idx + 10*SIMD_N_ELEM), ymm10);
        _mm256_store_pd(out + addrConv(rows, cols, idx + 11*SIMD_N_ELEM), ymm11);
        _mm256_store_pd(out + addrConv(rows, cols, idx + 12*SIMD_N_ELEM), ymm12);
        _mm256_store_pd(out + addrConv(rows, cols, idx + 13*SIMD_N_ELEM), ymm13);
        _mm256_store_pd(out + addrConv(rows, cols, idx + 14*SIMD_N_ELEM), ymm14);
        _mm256_store_pd(out + addrConv(rows, cols, idx + 15*SIMD_N_ELEM), ymm15);

        printf("read: %2d\twrite:%2d\n", (idx + 0 *SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 0 *SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 1 *SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 1 *SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 2 *SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 2 *SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 3 *SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 3 *SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 4 *SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 4 *SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 5 *SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 5 *SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 6 *SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 6 *SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 7 *SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 7 *SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 8 *SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 8 *SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 9 *SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 9 *SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 10*SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 10*SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 11*SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 11*SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 12*SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 12*SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 13*SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 13*SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 14*SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 14*SIMD_N_ELEM)/SIMD_N_ELEM);
        printf("read: %2d\twrite:%2d\n", (idx + 15*SIMD_N_ELEM)/SIMD_N_ELEM, addrConv(rows, cols, idx + 15*SIMD_N_ELEM)/SIMD_N_ELEM);
    }
}

/*
    Makes a dimension divisible by the smallest increment in which it can be read, that way columns and rows are properly memory aligned in an image without pow(n,2) columns
*/
inline int make_divisible
(
    int base,
    int min
) {
    // TODO
}