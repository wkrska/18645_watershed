#include <immintrin.h>
#include <stdint.h>
#include <omp.h>

#define DEBUG 0

#define C_BATCHES (4) // the number of SIMD reads per iteration
#define R_BATCHES (8) // the number of SIMD reads per iteration
#define D_WIDTH (8)
#define SIMD_N_ELEM (256/D_WIDTH) // the number of addresses to SIMD_N_ELEM for each SIMD read

// Function Prototypes
void mat_print(int cols, int rows, int8_t *mat);
inline void cols_kernel(int cols, int rows, int8_t* restrict in, int8_t* restrict out);
inline void rows_kernel(int cols, int rows,  int8_t* restrict in,  int8_t* restrict out);
inline void pack(int cols, int rows, int8_t* restrict in, int8_t* restrict out);
inline void unpack(int cols, int rows, int8_t* restrict in, int8_t* restrict out);


void mat_print(int cols, int rows, int8_t *mat) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%1d", (mat[j+i*cols]==-1) ? 1 : 0);
        }
        printf("\n");
    }
}

inline void cols_kernel
(
    int cols,
    int rows,
    int8_t* restrict in,
    int8_t* restrict out
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
        mat_print(rows*SIMD_N_ELEM,cols/SIMD_N_ELEM, out);
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
            mat_print(rows*SIMD_N_ELEM,cols/SIMD_N_ELEM, out);
            #endif
        }
        // // "Wind down" column
        // // Iteration 1
        // // Load values
        // LD0a =_mm256_setzero_pd();

        // // Perform AND for store
        // STR0 = (__m256d) _mm256_and_si256((__m256i) LD1b, (__m256i) CAR0);
        // STR1 = (__m256d) _mm256_and_si256((__m256i) LD2b, (__m256i) CAR1);
        // STR2 = (__m256d) _mm256_and_si256((__m256i) LD3b, (__m256i) CAR2);
        // STR3 = (__m256d) _mm256_and_si256((__m256i) LD0a, (__m256i) CAR3);

        // // Store values
        // _mm256_store_pd(out + (((rows/C_BATCHES)-1) * C_BATCHES + 0) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM), STR0);
        // _mm256_store_pd(out + (((rows/C_BATCHES)-1) * C_BATCHES + 1) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM), STR1);
        // _mm256_store_pd(out + (((rows/C_BATCHES)-1) * C_BATCHES + 2) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM), STR2);
        // _mm256_store_pd(out + (((rows/C_BATCHES)-1) * C_BATCHES + 3) * SIMD_N_ELEM + (c*rows*SIMD_N_ELEM), STR3);
    }
}

inline void rows_kernel
(
    int cols,
    int rows,
    int8_t* restrict in,
    int8_t* restrict out
) {
    volatile __m256d R0_0, R1_0,
                     R0_1, R1_1,
                     R0_2, R1_2,
                     R0_3, R1_3,
                     R0_4, R1_4,
                     R0_5, R1_5,
                     R0_6, R1_6,
                     R0_7, R1_7;

    #if DEBUG
    printf("\n\nDebugging Row Kernel\n\n");
    #endif

    // shifts "right", element 0 is zeroed
    //  LN0_0 = (__m256d) _mm256_alignr_epi8((__m256i) LN0_0, _mm256_permute2x128_si256((__m256i) LN0_0, (__m256i) LN0_0, _MM_SHUFFLE(0, 0, 2, 0)), 16 - 1);
    // shifts "left", rightmost element is zeroed
    //  LN0_0 = (__m256d) _mm256_alignr_epi8(_mm256_permute2x128_si256((__m256i) LN0_0, (__m256i) LN0_0, _MM_SHUFFLE(2, 0, 0, 1)), (__m256i) LN0_0, 1);

    // For every row
    // #pragma omp parallel for num_threads(4)
    for (int r = 0; r < rows; r++) {
        // For first SIMD load, need to do slow bit shift
        #if DEBUG
        printf("%d %d\n", r * cols ,r * cols + 1);
        #endif
        R0_5 = _mm256_load_pd(in + r * cols); // Aligned load
        R1_5 = (__m256d) _mm256_lddqu_si256(in + r * cols + 1); // Unligned load
        R0_7 = (__m256d) _mm256_permute2x128_si256((__m256i) R0_5, (__m256i) R0_5, _MM_SHUFFLE(0, 0, 2, 0)); // Step 1 of bitshift
        R1_7 = (__m256d) _mm256_and_si256((__m256i) R0_5, (__m256i) R1_5);
        R0_7 = (__m256d) _mm256_alignr_epi8((__m256i) R0_5, (__m256i) R0_7, 16 - 1); // Step 2 if bitshift RIGHT
        R0_7 = (__m256d) _mm256_and_si256((__m256i) R0_7, (__m256i) R1_7);
        _mm256_store_pd(out + r * cols, R0_7);

        #if DEBUG
        printf("first load\n");
        #endif

        // For every SIMD load of row besides first and last
        for (int c = 1; c < ((cols/SIMD_N_ELEM)-1); c+= R_BATCHES) {
            // Load first 2 vals, perform and, load next val, perform final and then store

            // Itr 1
            R0_0 = (__m256d) _mm256_lddqu_si256(in + r * cols + (c + 0) * SIMD_N_ELEM - 1);
            R1_0 = _mm256_load_pd(in + r * cols + (c + 0) * SIMD_N_ELEM);
            R0_0 = (__m256d) _mm256_and_si256((__m256i) R0_0, (__m256i) R1_0);
            R1_0 = (__m256d) _mm256_lddqu_si256(in + r * cols + (c + 0) * SIMD_N_ELEM + 1);
            R0_0 = (__m256d) _mm256_and_si256((__m256i) R0_0, (__m256i) R1_0);
            _mm256_store_pd(out + r * cols + (c + 0) * SIMD_N_ELEM, R0_0);

            // Itr 2
            R0_1 = (__m256d) _mm256_lddqu_si256(in + r * cols + (c + 0) * SIMD_N_ELEM - 1);
            R1_1 = _mm256_load_pd(in + r * cols + (c + 0) * SIMD_N_ELEM);
            R0_1 = (__m256d) _mm256_and_si256((__m256i) R0_1, (__m256i) R1_1);
            R1_1 = (__m256d) _mm256_lddqu_si256(in + r * cols + (c + 0) * SIMD_N_ELEM + 1);
            R0_1 = (__m256d) _mm256_and_si256((__m256i) R0_1, (__m256i) R1_1);
            _mm256_store_pd(out + r * cols + (c + 0) * SIMD_N_ELEM, R0_1);

            // Itr 3
            R0_2 = (__m256d) _mm256_lddqu_si256(in + r * cols + (c + 0) * SIMD_N_ELEM - 1);
            R1_2 = _mm256_load_pd(in + r * cols + (c + 0) * SIMD_N_ELEM);
            R0_2 = (__m256d) _mm256_and_si256((__m256i) R0_2, (__m256i) R1_2);
            R1_2 = (__m256d) _mm256_lddqu_si256(in + r * cols + (c + 0) * SIMD_N_ELEM + 1);
            R0_2 = (__m256d) _mm256_and_si256((__m256i) R0_2, (__m256i) R1_2);
            _mm256_store_pd(out + r * cols + (c + 0) * SIMD_N_ELEM, R0_2);

            // Itr 4
            R0_3 = (__m256d) _mm256_lddqu_si256(in + r * cols + (c + 0) * SIMD_N_ELEM - 1);
            R1_3 = _mm256_load_pd(in + r * cols + (c + 0) * SIMD_N_ELEM);
            R0_3 = (__m256d) _mm256_and_si256((__m256i) R0_3, (__m256i) R1_3);
            R1_3 = (__m256d) _mm256_lddqu_si256(in + r * cols + (c + 0) * SIMD_N_ELEM + 1);
            R0_3 = (__m256d) _mm256_and_si256((__m256i) R0_3, (__m256i) R1_3);
            _mm256_store_pd(out + r * cols + (c + 0) * SIMD_N_ELEM, R0_3);

            // Itr 5
            R0_4 = (__m256d) _mm256_lddqu_si256(in + r * cols + (c + 0) * SIMD_N_ELEM - 1);
            R1_4 = _mm256_load_pd(in + r * cols + (c + 0) * SIMD_N_ELEM);
            R0_4 = (__m256d) _mm256_and_si256((__m256i) R0_4, (__m256i) R1_4);
            R1_4 = (__m256d) _mm256_lddqu_si256(in + r * cols + (c + 0) * SIMD_N_ELEM + 1);
            R0_4 = (__m256d) _mm256_and_si256((__m256i) R0_4, (__m256i) R1_4);
            _mm256_store_pd(out + r * cols + (c + 0) * SIMD_N_ELEM, R0_4);

            // Itr 6
            R0_5 = (__m256d) _mm256_lddqu_si256(in + r * cols + (c + 0) * SIMD_N_ELEM - 1);
            R1_5 = _mm256_load_pd(in + r * cols + (c + 0) * SIMD_N_ELEM);
            R0_5 = (__m256d) _mm256_and_si256((__m256i) R0_5, (__m256i) R1_5);
            R1_5 = (__m256d) _mm256_lddqu_si256(in + r * cols + (c + 0) * SIMD_N_ELEM + 1);
            R0_5 = (__m256d) _mm256_and_si256((__m256i) R0_5, (__m256i) R1_5);
            _mm256_store_pd(out + r * cols + (c + 0) * SIMD_N_ELEM, R0_5);

            // Itr 7
            R0_6 = (__m256d) _mm256_lddqu_si256(in + r * cols + (c + 0) * SIMD_N_ELEM - 1);
            R1_6 = _mm256_load_pd(in + r * cols + (c + 0) * SIMD_N_ELEM);
            R0_6 = (__m256d) _mm256_and_si256((__m256i) R0_6, (__m256i) R1_6);
            R1_6 = (__m256d) _mm256_lddqu_si256(in + r * cols + (c + 0) * SIMD_N_ELEM + 1);
            R0_6 = (__m256d) _mm256_and_si256((__m256i) R0_6, (__m256i) R1_6);
            _mm256_store_pd(out + r * cols + (c + 0) * SIMD_N_ELEM, R0_6);

            // Itr 8
            #if DEBUG
            // printf("%d %d %d\n", r * cols + (c + 2) * SIMD_N_ELEM - 1, r * cols + (c + 2) * SIMD_N_ELEM, r * cols + (c + 2) * SIMD_N_ELEM + 1);
            #endif
            R0_7 = (__m256d) _mm256_lddqu_si256(in + r * cols + (c + 0) * SIMD_N_ELEM - 1);
            R1_7 = _mm256_load_pd(in + r * cols + (c + 0) * SIMD_N_ELEM);
            R0_7 = (__m256d) _mm256_and_si256((__m256i) R0_7, (__m256i) R1_7);
            R1_7 = (__m256d) _mm256_lddqu_si256(in + r * cols + (c + 0) * SIMD_N_ELEM + 1);
            R0_7 = (__m256d) _mm256_and_si256((__m256i) R0_7, (__m256i) R1_7);
            _mm256_store_pd(out + r * cols + (c + 0) * SIMD_N_ELEM, R0_7);

            
            #if DEBUG
            printf("loop load\n");
            #endif

        }

        // For last SIMD load
        // For first SIMD load, need to do slow bit shift
        #if DEBUG
        printf("%d %d\n", (r+1) * cols - SIMD_N_ELEM - 1, (r+1) * cols - SIMD_N_ELEM);
        #endif
        R0_0 = _mm256_load_pd(in + (r+1) * cols - SIMD_N_ELEM); // Aligned load
        R1_0 = (__m256d) _mm256_lddqu_si256(in + (r+1) * cols - SIMD_N_ELEM - 1); // Unligned load
        R0_1 = (__m256d) _mm256_permute2x128_si256((__m256i) R0_0, (__m256i) R0_0, _MM_SHUFFLE(2, 0, 0, 1)); // Step 1 of bitshift
        R1_1 = (__m256d) _mm256_and_si256((__m256i) R0_0, (__m256i) R1_0);
        R0_1 = (__m256d) _mm256_alignr_epi8((__m256i) R0_1, (__m256i) R0_0, 1); // Step 2 if bitshift LEFFT
        R0_1 = (__m256d) _mm256_and_si256((__m256i) R0_1, (__m256i) R1_1);
        _mm256_store_pd(out + (r+1) * cols - SIMD_N_ELEM, R0_1);
        
        #if DEBUG
        printf("last load\n");
        #endif

    }
}

/*
    Packs row major image, makes SIMD columns consecutive in memory
    Ex:
    [0 0 0 0] [1 1 1 1]     
    [2 2 2 2] [3 3 3 3] --> [0 0 0 0] [2 2 2 2] [4 4 4 4] [1 1 1 1] [3 3 3 3] [5 5 5 5]
    [4 4 4 4] [5 5 5 5]
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
