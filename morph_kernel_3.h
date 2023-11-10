#include <immintrin.h>
#include <stdint.h>

#define BATCH (4) // the number of SIMD reads per iteration
#define D_WIDTH (8)
#define STEP (256/D_WIDTH) // the number of addresses to step for each SIMD read

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

    // Load first BATCH rows
    ld0b = (__m256d) _mm256_lddqu_si256(in + 0 * STEP);
    ld1b = (__m256d) _mm256_lddqu_si256(in + 1 * STEP);
    ld2b = (__m256d) _mm256_lddqu_si256(in + 2 * STEP);
    ld3b = (__m256d) _mm256_lddqu_si256(in + 3 * STEP);
    
    // Perform AND for next round (car0 is 0)
    car1 = (__m256d) _mm256_and_si256((__m256i) ld1b, (__m256i) ld0b);
    car2 = (__m256d) _mm256_and_si256((__m256i) ld2b, (__m256i) ld1b);
    car3 = (__m256d) _mm256_and_si256((__m256i) ld3b, (__m256i) ld2b);

    // Rows are packed end to end, but first BATCH rows have been pre-processed to start the column. There will be rows/(rows read per itr) iterations
    for (int r = 1; r < rows/BATCH; r+=2) {
        // Iteration 1
        // Load values
        // Address = in + (iteration*number of reads per iteration+my_offset) * number of values per read 
        ld0a = (__m256d) _mm256_lddqu_si256(in + (r * BATCH + 0) * STEP);
        ld1a = (__m256d) _mm256_lddqu_si256(in + (r * BATCH + 1) * STEP);
        ld2a = (__m256d) _mm256_lddqu_si256(in + (r * BATCH + 2) * STEP);
        ld3a = (__m256d) _mm256_lddqu_si256(in + (r * BATCH + 3) * STEP);

        // Perform AND for store
        str0 = (__m256d) _mm256_and_si256((__m256i) ld1b, (__m256i) car0);
        str1 = (__m256d) _mm256_and_si256((__m256i) ld2b, (__m256i) car1);
        str2 = (__m256d) _mm256_and_si256((__m256i) ld3b, (__m256i) car2);
        str3 = (__m256d) _mm256_and_si256((__m256i) ld0a, (__m256i) car3);

        // Store values
        _mm256_store_pd(out + ((r-1) * BATCH + 0) * STEP , str0);
        _mm256_store_pd(out + ((r-1) * BATCH + 1) * STEP , str1);
        _mm256_store_pd(out + ((r-1) * BATCH + 2) * STEP , str2);
        _mm256_store_pd(out + ((r-1) * BATCH + 3) * STEP , str3);

        // Perform AND for next round
        car0 = (__m256d) _mm256_and_si256((__m256i) ld0a, (__m256i) ld3b);
        car1 = (__m256d) _mm256_and_si256((__m256i) ld1a, (__m256i) ld0a);
        car2 = (__m256d) _mm256_and_si256((__m256i) ld2a, (__m256i) ld1a);
        car3 = (__m256d) _mm256_and_si256((__m256i) ld3a, (__m256i) ld2a);

        // Iteration 2
        // Load values
        // Address = in + (iteration*number of reads per iteration+my_offset) * number of values per read 
        ld0b = (__m256d) _mm256_lddqu_si256(in + ((r+1) * BATCH + 0) * STEP);
        ld1b = (__m256d) _mm256_lddqu_si256(in + ((r+1) * BATCH + 1) * STEP);
        ld2b = (__m256d) _mm256_lddqu_si256(in + ((r+1) * BATCH + 2) * STEP);
        ld3b = (__m256d) _mm256_lddqu_si256(in + ((r+1) * BATCH + 3) * STEP);

        // Perform AND for store
        str0 = (__m256d) _mm256_and_si256((__m256i) ld1a, (__m256i) car0);
        str1 = (__m256d) _mm256_and_si256((__m256i) ld2a, (__m256i) car1);
        str2 = (__m256d) _mm256_and_si256((__m256i) ld3a, (__m256i) car2);
        str3 = (__m256d) _mm256_and_si256((__m256i) ld0b, (__m256i) car3);

        // Store values
        _mm256_store_pd(out + (r * BATCH + 0) * STEP , str0);
        _mm256_store_pd(out + (r * BATCH + 1) * STEP , str1);
        _mm256_store_pd(out + (r * BATCH + 2) * STEP , str2);
        _mm256_store_pd(out + (r * BATCH + 3) * STEP , str3);

        // Perform AND for next round
        car0 = (__m256d) _mm256_and_si256((__m256i) ld0b, (__m256i) ld3a);
        car1 = (__m256d) _mm256_and_si256((__m256i) ld1b, (__m256i) ld0b);
        car2 = (__m256d) _mm256_and_si256((__m256i) ld2b, (__m256i) ld1b);
        car3 = (__m256d) _mm256_and_si256((__m256i) ld3b, (__m256i) ld2b);
    }
}
