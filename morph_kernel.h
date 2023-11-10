#include <immintrin.h>
#include <stdint.h>

#define BATCH (4) // the number of SIMD reads per iteration
#define D_WIDTH (8)
#define STEP (256/D_WIDTH) // the number of addresses to step for each SIMD read

inline void morph_kernel
(
    int r,
    int8_t* restrict in,
    int8_t* restrict out
) {
    // To hold all new values
    __m256i ld0a, ld1a, ld2a, ld3a;
    __m256i ld0b, ld1b, ld2b, ld3b;
    // To hold all store values
    __m256i str0, str1, str2, str3;
    // To store all carry values
    __m256i car0, car1, car2, car3;

    // Zero values
    ld0a = _mm256_setzero_si256();
    ld1a = _mm256_setzero_si256();
    ld2a = _mm256_setzero_si256();
    ld3a = _mm256_setzero_si256();
    ld0b = _mm256_setzero_si256();
    ld1b = _mm256_setzero_si256();
    ld2b = _mm256_setzero_si256();
    ld3b = _mm256_setzero_si256();

    str0 = _mm256_setzero_si256();
    str1 = _mm256_setzero_si256();
    str2 = _mm256_setzero_si256();
    str3 = _mm256_setzero_si256();

    car0 = _mm256_setzero_si256();
    car1 = _mm256_setzero_si256();
    car2 = _mm256_setzero_si256();
    car3 = _mm256_setzero_si256();

    // "startup" before kernel

    // Load first BATCH rows
    ld0b = _mm256_load_si256((__m256i*) in + 0 * STEP);
    ld1b = _mm256_load_si256((__m256i*) in + 1 * STEP);
    ld2b = _mm256_load_si256((__m256i*) in + 2 * STEP);
    ld3b = _mm256_load_si256((__m256i*) in + 3 * STEP);
    
    // Perform AND for next round (car0 is 0)
    car1 = _mm256_and_si256(ld1b, ld0b);
    car2 = _mm256_and_si256(ld2b, ld1b);
    car3 = _mm256_and_si256(ld3b, ld2b);

    // Rows are packed end to end, but first BATCH rows have been pre-processed to start the column. There will be rows/(rows read per itr) iterations
    for (int i = 1; i < r/BATCH; i+=2) {
        // Iteration 1
        // Load values
        // Address = in + (iteration*number of reads per iteration+my_offset) * number of values per read 
        ld0a = _mm256_load_si256((__m256i*) in + (i * BATCH + 0) * STEP);
        ld1a = _mm256_load_si256((__m256i*) in + (i * BATCH + 1) * STEP);
        ld2a = _mm256_load_si256((__m256i*) in + (i * BATCH + 2) * STEP);
        ld3a = _mm256_load_si256((__m256i*) in + (i * BATCH + 3) * STEP);

        // Perform AND for store
        str0 = _mm256_and_si256(ld1b, car0);
        str1 = _mm256_and_si256(ld2b, car1);
        str2 = _mm256_and_si256(ld3b, car2);
        str3 = _mm256_and_si256(ld0a, car3);

        // Store values
        _mm256_store_si256((__m256i*) out + ((i-1) * BATCH + 0) * STEP , str0);
        _mm256_store_si256((__m256i*) out + ((i-1) * BATCH + 1) * STEP , str1);
        _mm256_store_si256((__m256i*) out + ((i-1) * BATCH + 2) * STEP , str2);
        _mm256_store_si256((__m256i*) out + ((i-1) * BATCH + 3) * STEP , str3);

        // Perform AND for next round
        car0 = _mm256_and_si256(ld0a, ld3b);
        car1 = _mm256_and_si256(ld1a, ld0a);
        car2 = _mm256_and_si256(ld2a, ld1a);
        car3 = _mm256_and_si256(ld3a, ld2a);

        // Iteration 2
        // Load values
        // Address = in + (iteration*number of reads per iteration+my_offset) * number of values per read 
        ld0b = _mm256_load_si256((__m256i*) in + ((i+1) * BATCH + 0) * STEP);
        ld1b = _mm256_load_si256((__m256i*) in + ((i+1) * BATCH + 1) * STEP);
        ld2b = _mm256_load_si256((__m256i*) in + ((i+1) * BATCH + 2) * STEP);
        ld3b = _mm256_load_si256((__m256i*) in + ((i+1) * BATCH + 3) * STEP);

        // Perform AND for store
        str0 = _mm256_and_si256(ld1a, car0);
        str1 = _mm256_and_si256(ld2a, car1);
        str2 = _mm256_and_si256(ld3a, car2);
        str3 = _mm256_and_si256(ld0b, car3);

        // Store values
        _mm256_store_si256((__m256i*) out + (i * BATCH + 0) * STEP , str0);
        _mm256_store_si256((__m256i*) out + (i * BATCH + 1) * STEP , str1);
        _mm256_store_si256((__m256i*) out + (i * BATCH + 2) * STEP , str2);
        _mm256_store_si256((__m256i*) out + (i * BATCH + 3) * STEP , str3);

        // Perform AND for next round
        car0 = _mm256_and_si256(ld0b, ld3a);
        car1 = _mm256_and_si256(ld1b, ld0b);
        car2 = _mm256_and_si256(ld2b, ld1b);
        car3 = _mm256_and_si256(ld3b, ld2b);
    }
}