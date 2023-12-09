#include <immintrin.h>
#include <stdint.h>
#include <omp.h>

#define DEBUG 0
// #define DEBUG_UNALIGNED

#define C_BATCHES (4) // the number of SIMD reads per iteration
#define R_BATCHES (4) // the number of SIMD reads per iteration
#define D_WIDTH (8)
#define SIMD_N_ELEM (256/D_WIDTH) // the number of addresses to SIMD_N_ELEM for each SIMD read

// Function Prototypes
//void mat_print(int cols, int rows, int8_t *mat);
// void cols_kernel(int cols, int rows, int8_t* /* restrict */ in, int8_t* /* restrict */ out);
// void rows_kernel(int cols, int rows,  int8_t* /* restrict */ in,  int8_t* /* restrict */ out);
// void pack(int cols, int rows, int8_t* /* restrict */ in, int8_t* /* restrict */ out);
// void unpack(int cols, int rows, int8_t* /* restrict */ in, int8_t* /* restrict */ out);


void mat_print(int cols, int rows, int8_t *mat) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%1d", (mat[j+i*cols]==-1) ? 1 : 0);
        }
        printf("\n");
    }
}

void cols_kernel_and
(
    int cols,
    int rows,
    int8_t* /* restrict */ in,
    int8_t* /* restrict */ out
) {
    // To hold all new values
    __m256d LD0a, LD1a, LD2a, LD3a;
    __m256d LD0b, LD1b, LD2b, LD3b;
    // To hold all store values
    __m256d STR0, STR1, STR2, STR3;
    // To store all CARry values
    __m256d CAR0, CAR1, CAR2, CAR3;

    #if DEBUG
    printf("\n\nDebugging Col Kernel\n\n");
    #endif

    // For all columns
    // #pragma omp parallel for num_threads(4)
    for (int c = 0; c < cols/SIMD_N_ELEM; c++) {
        // Zero values
        // LD0a = _mm256_setzero_pd();
        LD1a = _mm256_setzero_pd();
        LD2a = _mm256_setzero_pd();
        LD3a = _mm256_setzero_pd();
        LD0b = _mm256_setzero_pd();
        LD1b = _mm256_setzero_pd();
        LD2b = _mm256_setzero_pd();
        LD3b = _mm256_setzero_pd();

        STR0 = _mm256_setzero_pd();
        STR1 = _mm256_setzero_pd();
        STR2 = _mm256_setzero_pd();
        STR3 = _mm256_setzero_pd();

        CAR0 = _mm256_setzero_pd();
        CAR1 = _mm256_setzero_pd();
        CAR2 = _mm256_setzero_pd();
        CAR3 = _mm256_setzero_pd();

        // "startup" before kernel

        // Load first C_BATCHES rows
        LD0b = _mm256_load_pd(in + 0 * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));
        LD1b = _mm256_load_pd(in + 1 * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));
        LD2b = _mm256_load_pd(in + 2 * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));
        LD3b = _mm256_load_pd(in + 3 * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));
        
        // Perform AND for next round (CAR0 is 0)
        CAR1 = (__m256d) _mm256_and_si256((__m256i) LD1b, (__m256i) LD0b);
        CAR2 = (__m256d) _mm256_and_si256((__m256i) LD2b, (__m256i) LD1b);
        CAR3 = (__m256d) _mm256_and_si256((__m256i) LD3b, (__m256i) LD2b);

        #if DEBUG
        printf("c=0, r=0\n");
        // mat_print(rows*SIMD_N_ELEM,cols/SIMD_N_ELEM, out);
        #endif

        // Rows are packed end to end, but first C_BATCHES rows have been pre-processed to start the column. There will be rows/(rows read per itr) iterations
        for (int r = 1; r < rows/C_BATCHES; r+=2) {
            // Iteration 1
            // Load values
            // Address = in + (iteration*number of reads per iteration+my_offset) * number of values per read 
            LD0a = _mm256_load_pd(in + (r * C_BATCHES + 0) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));
            LD1a = _mm256_load_pd(in + (r * C_BATCHES + 1) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));
            LD2a = _mm256_load_pd(in + (r * C_BATCHES + 2) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));
            LD3a = _mm256_load_pd(in + (r * C_BATCHES + 3) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));

            // Perform AND for store
            STR0 = (__m256d) _mm256_and_si256((__m256i) LD1b, (__m256i) CAR0);
            STR1 = (__m256d) _mm256_and_si256((__m256i) LD2b, (__m256i) CAR1);
            STR2 = (__m256d) _mm256_and_si256((__m256i) LD3b, (__m256i) CAR2);
            STR3 = (__m256d) _mm256_and_si256((__m256i) LD0a, (__m256i) CAR3);

            // Store values
            _mm256_store_pd(out + ((r-1) * C_BATCHES + 0) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM), STR0);
            _mm256_store_pd(out + ((r-1) * C_BATCHES + 1) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM), STR1);
            _mm256_store_pd(out + ((r-1) * C_BATCHES + 2) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM), STR2);
            _mm256_store_pd(out + ((r-1) * C_BATCHES + 3) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM), STR3);

            // Perform AND for next round
            CAR0 = (__m256d) _mm256_and_si256((__m256i) LD0a, (__m256i) LD3b);
            CAR1 = (__m256d) _mm256_and_si256((__m256i) LD1a, (__m256i) LD0a);
            CAR2 = (__m256d) _mm256_and_si256((__m256i) LD2a, (__m256i) LD1a);
            CAR3 = (__m256d) _mm256_and_si256((__m256i) LD3a, (__m256i) LD2a);

            // Iteration 2
            // Load values
            // Address = in + (iteration*number of reads per iteration+my_offset) * number of values per read 
            LD0b = _mm256_load_pd(in + ((r+1) * C_BATCHES + 0) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));
            LD1b = _mm256_load_pd(in + ((r+1) * C_BATCHES + 1) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));
            LD2b = _mm256_load_pd(in + ((r+1) * C_BATCHES + 2) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));
            LD3b = _mm256_load_pd(in + ((r+1) * C_BATCHES + 3) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));

            // Perform AND for store
            STR0 = (__m256d) _mm256_and_si256((__m256i) LD1a, (__m256i) CAR0);
            STR1 = (__m256d) _mm256_and_si256((__m256i) LD2a, (__m256i) CAR1);
            STR2 = (__m256d) _mm256_and_si256((__m256i) LD3a, (__m256i) CAR2);
            STR3 = (__m256d) _mm256_and_si256((__m256i) LD0b, (__m256i) CAR3);

            // Store values
            _mm256_store_pd(out + (r * C_BATCHES + 0) * SIMD_N_ELEM  + (c*rows*SIMD_N_ELEM), STR0);
            _mm256_store_pd(out + (r * C_BATCHES + 1) * SIMD_N_ELEM  + (c*rows*SIMD_N_ELEM), STR1);
            _mm256_store_pd(out + (r * C_BATCHES + 2) * SIMD_N_ELEM  + (c*rows*SIMD_N_ELEM), STR2);
            _mm256_store_pd(out + (r * C_BATCHES + 3) * SIMD_N_ELEM  + (c*rows*SIMD_N_ELEM), STR3);

            // Perform AND for next round
            CAR0 = (__m256d) _mm256_and_si256((__m256i) LD0b, (__m256i) LD3a);
            CAR1 = (__m256d) _mm256_and_si256((__m256i) LD1b, (__m256i) LD0b);
            CAR2 = (__m256d) _mm256_and_si256((__m256i) LD2b, (__m256i) LD1b);
            CAR3 = (__m256d) _mm256_and_si256((__m256i) LD3b, (__m256i) LD2b);

            #if DEBUG
            printf("c=%d, r=%d\n", c, r);
            // uint8_t buff[rows*cols];
            // unpack(cols, rows, out, buff);
            // mat_print(cols,rows, buff);
            // mat_print(rows*SIMD_N_ELEM,cols/SIMD_N_ELEM, out);
            #endif
        }
    }
}

void cols_kernel_or
(
    int cols,
    int rows,
    int8_t* /* restrict */ in,
    int8_t* /* restrict */ out
) {
    // To hold all new values
    __m256d LD0a, LD1a, LD2a, LD3a;
    __m256d LD0b, LD1b, LD2b, LD3b;
    // To hold all store values
    __m256d STR0, STR1, STR2, STR3;
    // To store all CARry values
    __m256d CAR0, CAR1, CAR2, CAR3;

    #if DEBUG
    printf("\n\nDebugging Col Kernel\n\n");
    #endif

    // For all columns
    // #pragma omp parallel for num_threads(4)
    for (int c = 0; c < cols/SIMD_N_ELEM; c++) {
        // Zero values
        // LD0a = _mm256_setzero_pd();
        LD1a = _mm256_setzero_pd();
        LD2a = _mm256_setzero_pd();
        LD3a = _mm256_setzero_pd();
        LD0b = _mm256_setzero_pd();
        LD1b = _mm256_setzero_pd();
        LD2b = _mm256_setzero_pd();
        LD3b = _mm256_setzero_pd();

        STR0 = _mm256_setzero_pd();
        STR1 = _mm256_setzero_pd();
        STR2 = _mm256_setzero_pd();
        STR3 = _mm256_setzero_pd();

        CAR0 = _mm256_setzero_pd();
        CAR1 = _mm256_setzero_pd();
        CAR2 = _mm256_setzero_pd();
        CAR3 = _mm256_setzero_pd();

        // "startup" before kernel

        // Load first C_BATCHES rows
        LD0b = _mm256_load_pd(in + 0 * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));
        LD1b = _mm256_load_pd(in + 1 * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));
        LD2b = _mm256_load_pd(in + 2 * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));
        LD3b = _mm256_load_pd(in + 3 * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));
        
        // Perform AND for next round (CAR0 is 0)
        CAR1 = (__m256d) _mm256_and_si256((__m256i) LD1b, (__m256i) LD0b);
        CAR2 = (__m256d) _mm256_and_si256((__m256i) LD2b, (__m256i) LD1b);
        CAR3 = (__m256d) _mm256_and_si256((__m256i) LD3b, (__m256i) LD2b);

        #if DEBUG
        printf("c=0, r=0\n");
        // mat_print(rows*SIMD_N_ELEM,cols/SIMD_N_ELEM, out);
        #endif

        // Rows are packed end to end, but first C_BATCHES rows have been pre-processed to start the column. There will be rows/(rows read per itr) iterations
        for (int r = 1; r < rows/C_BATCHES; r+=2) {
            // Iteration 1
            // Load values
            // Address = in + (iteration*number of reads per iteration+my_offset) * number of values per read 
            LD0a = _mm256_load_pd(in + (r * C_BATCHES + 0) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));
            LD1a = _mm256_load_pd(in + (r * C_BATCHES + 1) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));
            LD2a = _mm256_load_pd(in + (r * C_BATCHES + 2) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));
            LD3a = _mm256_load_pd(in + (r * C_BATCHES + 3) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));

            // Perform AND for store
            STR0 = (__m256d) _mm256_or_si256((__m256i) LD1b, (__m256i) CAR0);
            STR1 = (__m256d) _mm256_or_si256((__m256i) LD2b, (__m256i) CAR1);
            STR2 = (__m256d) _mm256_or_si256((__m256i) LD3b, (__m256i) CAR2);
            STR3 = (__m256d) _mm256_or_si256((__m256i) LD0a, (__m256i) CAR3);

            // Store values
            _mm256_store_pd(out + ((r-1) * C_BATCHES + 0) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM), STR0);
            _mm256_store_pd(out + ((r-1) * C_BATCHES + 1) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM), STR1);
            _mm256_store_pd(out + ((r-1) * C_BATCHES + 2) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM), STR2);
            _mm256_store_pd(out + ((r-1) * C_BATCHES + 3) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM), STR3);

            // Perform AND for next round
            CAR0 = (__m256d) _mm256_or_si256((__m256i) LD0a, (__m256i) LD3b);
            CAR1 = (__m256d) _mm256_or_si256((__m256i) LD1a, (__m256i) LD0a);
            CAR2 = (__m256d) _mm256_or_si256((__m256i) LD2a, (__m256i) LD1a);
            CAR3 = (__m256d) _mm256_or_si256((__m256i) LD3a, (__m256i) LD2a);

            // Iteration 2
            // Load values
            // Address = in + (iteration*number of reads per iteration+my_offset) * number of values per read 
            LD0b = _mm256_load_pd(in + ((r+1) * C_BATCHES + 0) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));
            LD1b = _mm256_load_pd(in + ((r+1) * C_BATCHES + 1) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));
            LD2b = _mm256_load_pd(in + ((r+1) * C_BATCHES + 2) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));
            LD3b = _mm256_load_pd(in + ((r+1) * C_BATCHES + 3) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM));

            // Perform AND for store
            STR0 = (__m256d) _mm256_or_si256((__m256i) LD1a, (__m256i) CAR0);
            STR1 = (__m256d) _mm256_or_si256((__m256i) LD2a, (__m256i) CAR1);
            STR2 = (__m256d) _mm256_or_si256((__m256i) LD3a, (__m256i) CAR2);
            STR3 = (__m256d) _mm256_or_si256((__m256i) LD0b, (__m256i) CAR3);

            // Store values
            _mm256_store_pd(out + (r * C_BATCHES + 0) * SIMD_N_ELEM  + (c*rows*SIMD_N_ELEM), STR0);
            _mm256_store_pd(out + (r * C_BATCHES + 1) * SIMD_N_ELEM  + (c*rows*SIMD_N_ELEM), STR1);
            _mm256_store_pd(out + (r * C_BATCHES + 2) * SIMD_N_ELEM  + (c*rows*SIMD_N_ELEM), STR2);
            _mm256_store_pd(out + (r * C_BATCHES + 3) * SIMD_N_ELEM  + (c*rows*SIMD_N_ELEM), STR3);

            // Perform AND for next round
            CAR0 = (__m256d) _mm256_or_si256((__m256i) LD0b, (__m256i) LD3a);
            CAR1 = (__m256d) _mm256_or_si256((__m256i) LD1b, (__m256i) LD0b);
            CAR2 = (__m256d) _mm256_or_si256((__m256i) LD2b, (__m256i) LD1b);
            CAR3 = (__m256d) _mm256_or_si256((__m256i) LD3b, (__m256i) LD2b);

            #if DEBUG
            printf("c=%d, r=%d\n", c, r);
            // uint8_t buff[rows*cols];
            // unpack(cols, rows, out, buff);
            // mat_print(cols,rows, buff);
            // mat_print(rows*SIMD_N_ELEM,cols/SIMD_N_ELEM, out);
            #endif
        }
    }
}

void rows_kernel_and
(
    int cols,
    int rows,
    int8_t* /* restrict */ in,
    int8_t* /* restrict */ out
) {
     __m256d ymm0, ymm4, ymm8, ymm12,
             ymm1, ymm5, ymm9, ymm13,
             ymm2, ymm6, ymm10, ymm14,
             ymm3, ymm7, ymm11, ymm15;

    #if DEBUG
    printf("\n\nDebugging Row Kernel\n\n");
    #endif

    // For every row
    // #pragma omp parallel for num_threads(4)
    for (int r = 0; r < rows; r++) {
        // For first SIMD load, need to do slow bit shift
        ymm8 = _mm256_load_pd(in + r * cols); // Aligned load
        ymm9 = _mm256_loadu_pd(in + r * cols + 1); // Unligned load
        ymm10 = (__m256d) _mm256_permute2x128_si256((__m256i) ymm8, (__m256i) ymm8, _MM_SHUFFLE(0, 0, 2, 0)); // Step 1 of bitshift
        ymm11 = (__m256d) _mm256_and_si256((__m256i) ymm8, (__m256i) ymm9);
        ymm8 = (__m256d) _mm256_alignr_epi8((__m256i) ymm8, (__m256i) ymm10, 16 - 1); // Step 2 if bitshift RIGHT
        ymm9 = (__m256d) _mm256_and_si256((__m256i) ymm8, (__m256i) ymm11);
        _mm256_store_pd(out + r * cols, ymm9);

        // Precompute pointers
        uint8_t* ptr0 = in + r * cols;
        uint8_t* ptr1 = in + r * cols;
        uint8_t* ptr2 = in + r * cols;
        uint8_t* ptr3 = in + r * cols;

        // For every SIMD load of row besides first and last
        for (int c = 1; c < ((cols/SIMD_N_ELEM)-1); c+= R_BATCHES) {
            // Load first 2 vals, perform and, load next val, perform final and then store
            uint8_t* cptr0 = ptr0 + (c + 0) * SIMD_N_ELEM;
            uint8_t* cptr1 = ptr1 + (c + 1) * SIMD_N_ELEM;
            uint8_t* cptr2 = ptr2 + (c + 2) * SIMD_N_ELEM;
            uint8_t* cptr3 = ptr3 + (c + 3) * SIMD_N_ELEM;

            // Itr 1
            #if R_BATCHES >= 1
            #ifndef DEBUG_UNALIGNED
            ymm0 = _mm256_loadu_pd(cptr0 - 1);
            #else
            ymm0 = _mm256_load_pd(cptr0);
            #endif
            ymm1 = _mm256_load_pd(cptr0);
            #ifndef DEBUG_UNALIGNED
            ymm2 = _mm256_loadu_pd(cptr0 + 1);
            #else
            ymm2 = _mm256_load_pd(cptr0);
            #endif
            ymm3 = (__m256d) _mm256_and_si256((__m256i) ymm0, (__m256i) ymm1);
            ymm0 = (__m256d) _mm256_and_si256((__m256i) ymm2, (__m256i) ymm3);
            _mm256_store_pd(out + r * cols + (c + 0) * SIMD_N_ELEM, ymm0);
            #endif
            // Itr 2
            #if R_BATCHES >= 2
            #ifndef DEBUG_UNALIGNED
            ymm4 = _mm256_loadu_pd(cptr1 - 1);
            #else
            ymm4 = _mm256_load_pd(cptr1);
            #endif
            ymm5 = _mm256_load_pd(cptr1);
            #ifndef DEBUG_UNALIGNED
            ymm6 = _mm256_loadu_pd(cptr1 + 1);
            #else
            ymm6 = _mm256_load_pd(cptr1);
            #endif
            ymm7 = (__m256d) _mm256_and_si256((__m256i) ymm4, (__m256i) ymm5);
            ymm4 = (__m256d) _mm256_and_si256((__m256i) ymm6, (__m256i) ymm7);
            _mm256_store_pd(out + r * cols + (c + 1) * SIMD_N_ELEM, ymm4);
            #endif
            // Itr 3
            #if R_BATCHES >= 3
            #ifndef DEBUG_UNALIGNED
            ymm8 = _mm256_loadu_pd(cptr2 - 1);
            #else
            ymm8 = _mm256_load_pd(cptr2);
            #endif
            ymm9 = _mm256_load_pd(cptr2);
            #ifndef DEBUG_UNALIGNED
            ymm10 = _mm256_loadu_pd(cptr2 + 1);
            #else
            ymm10 = _mm256_load_pd(cptr2);
            #endif
            ymm11 = (__m256d) _mm256_and_si256((__m256i) ymm8, (__m256i) ymm9);
            ymm8 = (__m256d) _mm256_and_si256((__m256i) ymm10, (__m256i) ymm11);
            _mm256_store_pd(out + r * cols + (c + 2) * SIMD_N_ELEM, ymm8);
            #endif
            // Itr 4
            #if R_BATCHES >= 4
            #ifndef DEBUG_UNALIGNED
            ymm12 = _mm256_loadu_pd(cptr3 - 1);
            #else
            ymm12 = _mm256_load_pd(cptr3);
            #endif
            ymm13 = _mm256_load_pd(cptr3);
            #ifndef DEBUG_UNALIGNED
            ymm14 = _mm256_loadu_pd(cptr3 + 1);
            #else
            ymm14 = _mm256_load_pd(cptr3);
            #endif
            ymm15 = (__m256d) _mm256_and_si256((__m256i) ymm12, (__m256i) ymm13);
            ymm12 = (__m256d) _mm256_and_si256((__m256i) ymm14, (__m256i) ymm15);
            _mm256_store_pd(out + r * cols + (c + 3) * SIMD_N_ELEM, ymm12);
            #endif

            #if DEBUG
            printf("Row %d Col%d\n",r,c);
            #endif

        }

        // For last SIMD load
        // For first SIMD load, need to do slow bit shift
        #if DEBUG
        // printf("%d %d\n", (r+1) * cols - SIMD_N_ELEM - 1, (r+1) * cols - SIMD_N_ELEM);
        #endif
        ymm0 = _mm256_load_pd(in + (r+1) * cols - SIMD_N_ELEM); // Aligned load
        ymm1 = _mm256_loadu_pd(in + (r+1) * cols - SIMD_N_ELEM - 1); // Unligned load
        ymm2 = (__m256d) _mm256_permute2x128_si256((__m256i) ymm0, (__m256i) ymm0, _MM_SHUFFLE(2, 0, 0, 1)); // Step 1 of bitshift
        ymm3 = (__m256d) _mm256_and_si256((__m256i) ymm0, (__m256i) ymm1);
        ymm0 = (__m256d) _mm256_alignr_epi8((__m256i) ymm2, (__m256i) ymm0, 1); // Step 2 if bitshift LEFFT
        ymm1 = (__m256d) _mm256_and_si256((__m256i) ymm0, (__m256i) ymm3);
        _mm256_store_pd(out + (r+1) * cols - SIMD_N_ELEM, ymm1);
    }
}

/*
    Packs row major image, makes SIMD columns consecutive in memory
    Ex:
    [0 0 0 0] [1 1 1 1]     
    [2 2 2 2] [3 3 3 3] --> [0 0 0 0] [2 2 2 2] [4 4 4 4] [1 1 1 1] [3 3 3 3] [5 5 5 5]
    [4 4 4 4] [5 5 5 5]
*/
void pack
(
    int cols,
    int rows,
    int8_t* /* restrict */ in,
    int8_t* /* restrict */ out
) {
     int addrConv(int rows, int cols, int addr){
        // Read addr: row*elements in full row + col*elements in SIMD read
        // Write addr: row*elements in SIMD read + col*elements in full SIMD column
        int row = addr/cols;
        int col = addr%cols;
        return row*SIMD_N_ELEM + col*rows;
    }

    #if DEBUG
    printf("\n\nDebugging Packing Kernel\n\n");
    printf("Rows: %d\nCols: %d\nSIMD_N_ELEN: %d\n\n", rows, cols,SIMD_N_ELEM);
    #endif

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

        #if DEBUG
        printf("read: %2d\twrite:%2d\n", (idx + 0 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 0 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 1 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 1 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 2 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 2 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 3 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 3 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 4 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 4 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 5 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 5 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 6 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 6 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 7 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 7 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 8 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 8 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 9 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 9 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 10*SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 10*SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 11*SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 11*SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 12*SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 12*SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 13*SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 13*SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 14*SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 14*SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 15*SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 15*SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        #endif
    }
}

/*
    Unpacks consecutive SIMD columns into row major image
    Ex: 
                                                                    [0 0 0 0] [1 1 1 1]
    [0 0 0 0] [2 2 2 2] [4 4 4 4] [1 1 1 1] [3 3 3 3] [5 5 5 5] --> [2 2 2 2] [3 3 3 3]
                                                                    [4 4 4 4] [5 5 5 5]
*/
void unpack
(
    int cols,
    int rows,
    int8_t* /* restrict */ in,
    int8_t* /* restrict */ out
) {
     int addrConv(int rows, int cols, int addr){
        // Read addr: row*elements in full row + col*elements in SIMD read
        // Write addr: row*elements in SIMD read + col*elements in full SIMD column
        int row = (addr/SIMD_N_ELEM)%rows;
        int col = addr/(rows*SIMD_N_ELEM)*SIMD_N_ELEM; /// seems redundant, need it for integer division!
        return row*cols + col;
    }

    #if DEBUG
    printf("\n\nDebugging Unpacking Kernel\n\n");
    printf("Rows: %d\nCols: %d\nSIMD_N_ELEN: %d\n\n", rows, cols,SIMD_N_ELEM);
    #endif

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

        #if DEBUG
        printf("read: %2d\twrite:%2d\n", (idx + 0 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 0 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 1 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 1 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 2 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 2 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 3 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 3 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 4 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 4 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 5 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 5 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 6 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 6 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 7 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 7 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 8 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 8 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 9 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 9 *SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 10*SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 10*SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 11*SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 11*SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 12*SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 12*SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 13*SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 13*SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 14*SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 14*SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        printf("read: %2d\twrite:%2d\n", (idx + 15*SIMD_N_ELEM)/*/SIMD_N_ELEM*/, addrConv(rows, cols, idx + 15*SIMD_N_ELEM)/*/SIMD_N_ELEM*/);
        #endif
    }
}
