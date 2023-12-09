void rows_kernel_alt_0
(
    int cols,
    int rows,
    int8_t* /* restrict */ in,
    int8_t* /* restrict */ out
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
        R0_5 = _mm256_load_pd(in + r * cols); // Aligned load
        R1_5 = _mm256_loadu_pd(in + r * cols + 1); // Unligned load
        R0_7 = (__m256d) _mm256_permute2x128_si256((__m256i) R0_5, (__m256i) R0_5, _MM_SHUFFLE(0, 0, 2, 0)); // Step 1 of bitshift
        R1_7 = (__m256d) _mm256_and_si256((__m256i) R0_5, (__m256i) R1_5);
        R0_7 = (__m256d) _mm256_alignr_epi8((__m256i) R0_5, (__m256i) R0_7, 16 - 1); // Step 2 if bitshift RIGHT
        R0_7 = (__m256d) _mm256_and_si256((__m256i) R0_7, (__m256i) R1_7);
        _mm256_store_pd(out + r * cols, R0_7);


        // For every SIMD load of row besides first and last
        for (int c = 1; c < ((cols/SIMD_N_ELEM)-1); c+= R_BATCHES) {
            // Load first 2 vals, perform and, load next val, perform final and then store

            // Itr 1
            #if R_BATCHES >= 1
            #ifndef DEBUG_UNALIGNED
            R0_0 = _mm256_loadu_pd(in + r * cols + (c + 0) * SIMD_N_ELEM - 1);
            #else
            R0_0 = _mm256_load_pd(in + r * cols + (c + 0) * SIMD_N_ELEM);
            #endif
            R1_0 = _mm256_load_pd(in + r * cols + (c + 0) * SIMD_N_ELEM);
            R0_0 = (__m256d) _mm256_and_si256((__m256i) R0_0, (__m256i) R1_0);
            #ifndef DEBUG_UNALIGNED
            R1_0 = _mm256_loadu_pd(in + r * cols + (c + 0) * SIMD_N_ELEM + 1);
            #else
            R1_0 = _mm256_load_pd(in + r * cols + (c + 0) * SIMD_N_ELEM);
            #endif
            R0_0 = (__m256d) _mm256_and_si256((__m256i) R0_0, (__m256i) R1_0);
            _mm256_store_pd(out + r * cols + (c + 0) * SIMD_N_ELEM, R0_0);
            #endif

            // Itr 2
            #if R_BATCHES >= 2
            #ifndef DEBUG_UNALIGNED
            R0_1 = _mm256_loadu_pd(in + r * cols + (c + 1) * SIMD_N_ELEM - 1);
            #else
            R0_1 = _mm256_load_pd(in + r * cols + (c + 1) * SIMD_N_ELEM);
            #endif
            R1_1 = _mm256_load_pd(in + r * cols + (c + 1) * SIMD_N_ELEM);
            R0_1 = (__m256d) _mm256_and_si256((__m256i) R0_1, (__m256i) R1_1);
            #ifndef DEBUG_UNALIGNED
            R1_1 = _mm256_loadu_pd(in + r * cols + (c + 1) * SIMD_N_ELEM + 1);
            #else
            R1_1 = _mm256_load_pd(in + r * cols + (c + 1) * SIMD_N_ELEM);
            #endif
            R0_1 = (__m256d) _mm256_and_si256((__m256i) R0_1, (__m256i) R1_1);
            _mm256_store_pd(out + r * cols + (c + 1) * SIMD_N_ELEM, R0_1);
            #endif

            // Itr 3
            #if R_BATCHES >= 3
            #ifndef DEBUG_UNALIGNED
            R0_2 = _mm256_loadu_pd(in + r * cols + (c + 2) * SIMD_N_ELEM - 1);
            #else
            R0_2 = _mm256_load_pd(in + r * cols + (c + 2) * SIMD_N_ELEM);
            #endif
            R1_2 = _mm256_load_pd(in + r * cols + (c + 2) * SIMD_N_ELEM);
            R0_2 = (__m256d) _mm256_and_si256((__m256i) R0_2, (__m256i) R1_2);
            #ifndef DEBUG_UNALIGNED
            R1_2 = _mm256_loadu_pd(in + r * cols + (c + 2) * SIMD_N_ELEM + 1);
            #else
            R1_2 = _mm256_load_pd(in + r * cols + (c + 2) * SIMD_N_ELEM);
            #endif
            R0_2 = (__m256d) _mm256_and_si256((__m256i) R0_2, (__m256i) R1_2);
            _mm256_store_pd(out + r * cols + (c + 2) * SIMD_N_ELEM, R0_2);
            #endif

            // Itr 4
            #if R_BATCHES >= 4
            #ifndef DEBUG_UNALIGNED
            R0_3 = _mm256_loadu_pd(in + r * cols + (c + 3) * SIMD_N_ELEM - 1);
            #else
            R0_3 = _mm256_load_pd(in + r * cols + (c + 3) * SIMD_N_ELEM);
            #endif
            R1_3 = _mm256_load_pd(in + r * cols + (c + 3) * SIMD_N_ELEM);
            R0_3 = (__m256d) _mm256_and_si256((__m256i) R0_3, (__m256i) R1_3);
            #ifndef DEBUG_UNALIGNED
            R1_3 = _mm256_loadu_pd(in + r * cols + (c + 3) * SIMD_N_ELEM + 1);
            #else
            R1_3 = _mm256_load_pd(in + r * cols + (c + 3) * SIMD_N_ELEM);
            #endif
            R0_3 = (__m256d) _mm256_and_si256((__m256i) R0_3, (__m256i) R1_3);
            _mm256_store_pd(out + r * cols + (c + 3) * SIMD_N_ELEM, R0_3);
            #endif

            // Itr 5
            #if R_BATCHES >= 5
            #ifndef DEBUG_UNALIGNED
            R0_4 = _mm256_loadu_pd(in + r * cols + (c + 4) * SIMD_N_ELEM - 1);
            #else
            R0_4 = _mm256_load_pd(in + r * cols + (c + 4) * SIMD_N_ELEM);
            #endif
            R1_4 = _mm256_load_pd(in + r * cols + (c + 4) * SIMD_N_ELEM);
            R0_4 = (__m256d) _mm256_and_si256((__m256i) R0_4, (__m256i) R1_4);
            #ifndef DEBUG_UNALIGNED
            R1_4 = _mm256_loadu_pd(in + r * cols + (c + 4) * SIMD_N_ELEM + 1);
            #else
            R1_4 = _mm256_load_pd(in + r * cols + (c + 4) * SIMD_N_ELEM);
            #endif
            R0_4 = (__m256d) _mm256_and_si256((__m256i) R0_4, (__m256i) R1_4);
            _mm256_store_pd(out + r * cols + (c + 4) * SIMD_N_ELEM, R0_4);
            #endif

            // Itr 6
            #if R_BATCHES >= 6
            #ifndef DEBUG_UNALIGNED
            R0_5 = _mm256_loadu_pd(in + r * cols + (c + 5) * SIMD_N_ELEM - 1);
            #else
            R0_5 = _mm256_load_pd(in + r * cols + (c + 5) * SIMD_N_ELEM);
            #endif
            R1_5 = _mm256_load_pd(in + r * cols + (c + 5) * SIMD_N_ELEM);
            R0_5 = (__m256d) _mm256_and_si256((__m256i) R0_5, (__m256i) R1_5);
            #ifndef DEBUG_UNALIGNED
            R1_5 = _mm256_loadu_pd(in + r * cols + (c + 5) * SIMD_N_ELEM + 1);
            #else
            R1_5 = _mm256_load_pd(in + r * cols + (c + 5) * SIMD_N_ELEM);
            #endif
            R0_5 = (__m256d) _mm256_and_si256((__m256i) R0_5, (__m256i) R1_5);
            _mm256_store_pd(out + r * cols + (c + 5) * SIMD_N_ELEM, R0_5);
            #endif

            // Itr 7
            #if R_BATCHES >= 7
            #ifndef DEBUG_UNALIGNED
            R0_6 = _mm256_loadu_pd(in + r * cols + (c + 6) * SIMD_N_ELEM - 1);
            #else
            R0_6 = _mm256_load_pd(in + r * cols + (c + 6) * SIMD_N_ELEM);
            #endif
            R1_6 = _mm256_load_pd(in + r * cols + (c + 6) * SIMD_N_ELEM);
            R0_6 = (__m256d) _mm256_and_si256((__m256i) R0_6, (__m256i) R1_6);
            #ifndef DEBUG_UNALIGNED
            R1_6 = _mm256_loadu_pd(in + r * cols + (c + 6) * SIMD_N_ELEM + 1);
            #else
            R1_6 = _mm256_load_pd(in + r * cols + (c + 6) * SIMD_N_ELEM);
            #endif
            R0_6 = (__m256d) _mm256_and_si256((__m256i) R0_6, (__m256i) R1_6);
            _mm256_store_pd(out + r * cols + (c + 6) * SIMD_N_ELEM, R0_6);
            #endif

            // Itr 8
            #if R_BATCHES >= 8
            #ifndef DEBUG_UNALIGNED
            R0_7 = _mm256_loadu_pd(in + r * cols + (c + 7) * SIMD_N_ELEM - 1);
            #else
            R0_7 = _mm256_load_pd(in + r * cols + (c + 7) * SIMD_N_ELEM);
            #endif
            R1_7 = _mm256_load_pd(in + r * cols + (c + 7) * SIMD_N_ELEM);
            R0_7 = (__m256d) _mm256_and_si256((__m256i) R0_7, (__m256i) R1_7);
            #ifndef DEBUG_UNALIGNED
            R1_7 = _mm256_loadu_pd(in + r * cols + (c + 7) * SIMD_N_ELEM + 1);
            #else
            R1_7 = _mm256_load_pd(in + r * cols + (c + 7) * SIMD_N_ELEM);
            #endif
            R0_7 = (__m256d) _mm256_and_si256((__m256i) R0_7, (__m256i) R1_7);
            _mm256_store_pd(out + r * cols + (c + 7) * SIMD_N_ELEM, R0_7);
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
        R0_0 = _mm256_load_pd(in + (r+1) * cols - SIMD_N_ELEM); // Aligned load
        R1_0 = _mm256_loadu_pd(in + (r+1) * cols - SIMD_N_ELEM - 1); // Unligned load
        R0_1 = (__m256d) _mm256_permute2x128_si256((__m256i) R0_0, (__m256i) R0_0, _MM_SHUFFLE(2, 0, 0, 1)); // Step 1 of bitshift
        R1_1 = (__m256d) _mm256_and_si256((__m256i) R0_0, (__m256i) R1_0);
        R0_1 = (__m256d) _mm256_alignr_epi8((__m256i) R0_1, (__m256i) R0_0, 1); // Step 2 if bitshift LEFFT
        R0_1 = (__m256d) _mm256_and_si256((__m256i) R0_1, (__m256i) R1_1);
        _mm256_store_pd(out + (r+1) * cols - SIMD_N_ELEM, R0_1);

    }
}

void rows_kernel_alt_1
(
    int cols,
    int rows,
    int8_t* /* restrict */ in,
    int8_t* /* restrict */ out
) {
     __m256d R0_0, R1_0, R2_0, R3_0,
             R0_1, R1_1, R2_1, R3_1,
             R0_2, R1_2, R2_2, R3_2,
             R0_3, R1_3, R2_3, R3_3;

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
        // // For first SIMD load, need to do slow bit shift
        // R2_0 = _mm256_load_pd(in + r * cols); // Aligned load
        // R2_1 = _mm256_loadu_pd(in + r * cols + 1); // Unligned load
        // R2_2 = (__m256d) _mm256_permute2x128_si256((__m256i) R2_0, (__m256i) R2_0, _MM_SHUFFLE(0, 0, 2, 0)); // Step 1 of bitshift
        // R2_3 = (__m256d) _mm256_and_si256((__m256i) R2_0, (__m256i) R2_1);
        // R2_0 = (__m256d) _mm256_alignr_epi8((__m256i) R2_0, (__m256i) R2_2, 16 - 1); // Step 2 if bitshift RIGHT
        // R2_1 = (__m256d) _mm256_and_si256((__m256i) R2_0, (__m256i) R2_3);
        // _mm256_store_pd(out + r * cols, R2_1);

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
            R0_0 = _mm256_loadu_pd(cptr0 - 1);
            #else
            R0_0 = _mm256_load_pd(cptr0);
            #endif
            R0_1 = _mm256_load_pd(cptr0);
            #ifndef DEBUG_UNALIGNED
            R0_2 = _mm256_loadu_pd(cptr0 + 1);
            #else
            R0_2 = _mm256_load_pd(cptr0);
            #endif
            R0_3 = (__m256d) _mm256_and_si256((__m256i) R0_0, (__m256i) R0_1);
            R0_0 = (__m256d) _mm256_and_si256((__m256i) R0_2, (__m256i) R0_3);
            _mm256_store_pd(out + r * cols + (c + 0) * SIMD_N_ELEM, R0_0);
            #endif
            // Itr 2
            #if R_BATCHES >= 2
            #ifndef DEBUG_UNALIGNED
            R1_0 = _mm256_loadu_pd(cptr1 - 1);
            #else
            R1_0 = _mm256_load_pd(cptr1);
            #endif
            R1_1 = _mm256_load_pd(cptr1);
            #ifndef DEBUG_UNALIGNED
            R1_2 = _mm256_loadu_pd(cptr1 + 1);
            #else
            R1_2 = _mm256_load_pd(cptr1);
            #endif
            R1_3 = (__m256d) _mm256_and_si256((__m256i) R1_0, (__m256i) R1_1);
            R1_0 = (__m256d) _mm256_and_si256((__m256i) R1_2, (__m256i) R1_3);
            _mm256_store_pd(out + r * cols + (c + 1) * SIMD_N_ELEM, R1_0);
            #endif
            // Itr 3
            #if R_BATCHES >= 3
            #ifndef DEBUG_UNALIGNED
            R2_0 = _mm256_loadu_pd(cptr2 - 1);
            #else
            R2_0 = _mm256_load_pd(cptr2);
            #endif
            R2_1 = _mm256_load_pd(cptr2);
            #ifndef DEBUG_UNALIGNED
            R2_2 = _mm256_loadu_pd(cptr2 + 1);
            #else
            R2_2 = _mm256_load_pd(cptr2);
            #endif
            R2_3 = (__m256d) _mm256_and_si256((__m256i) R2_0, (__m256i) R2_1);
            R2_0 = (__m256d) _mm256_and_si256((__m256i) R2_2, (__m256i) R2_3);
            _mm256_store_pd(out + r * cols + (c + 2) * SIMD_N_ELEM, R2_0);
            #endif
            // Itr 4
            #if R_BATCHES >= 4
            #ifndef DEBUG_UNALIGNED
            R3_0 = _mm256_loadu_pd(cptr3 - 1);
            #else
            R3_0 = _mm256_load_pd(cptr3);
            #endif
            R3_1 = _mm256_load_pd(cptr3);
            #ifndef DEBUG_UNALIGNED
            R3_2 = _mm256_loadu_pd(cptr3 + 1);
            #else
            R3_2 = _mm256_load_pd(cptr3);
            #endif
            R3_3 = (__m256d) _mm256_and_si256((__m256i) R3_0, (__m256i) R3_1);
            R3_0 = (__m256d) _mm256_and_si256((__m256i) R3_2, (__m256i) R3_3);
            _mm256_store_pd(out + r * cols + (c + 3) * SIMD_N_ELEM, R3_0);
            #endif

            #if DEBUG
            printf("Row %d Col%d\n",r,c);
            #endif

        }

        // // For last SIMD load
        // // For first SIMD load, need to do slow bit shift
        // #if DEBUG
        // // printf("%d %d\n", (r+1) * cols - SIMD_N_ELEM - 1, (r+1) * cols - SIMD_N_ELEM);
        // #endif
        // R0_0 = _mm256_load_pd(in + (r+1) * cols - SIMD_N_ELEM); // Aligned load
        // R0_1 = _mm256_loadu_pd(in + (r+1) * cols - SIMD_N_ELEM - 1); // Unligned load
        // R0_2 = (__m256d) _mm256_permute2x128_si256((__m256i) R0_0, (__m256i) R0_0, _MM_SHUFFLE(2, 0, 0, 1)); // Step 1 of bitshift
        // R0_3 = (__m256d) _mm256_and_si256((__m256i) R0_0, (__m256i) R0_1);
        // R0_0 = (__m256d) _mm256_alignr_epi8((__m256i) R0_2, (__m256i) R0_0, 1); // Step 2 if bitshift LEFFT
        // R0_1 = (__m256d) _mm256_and_si256((__m256i) R0_0, (__m256i) R0_3);
        // _mm256_store_pd(out + (r+1) * cols - SIMD_N_ELEM, R0_1);
    }
}
void rows_kernel_alt_2
(
    int cols,
    int rows,
    int8_t* /* restrict */ in,
    int8_t* /* restrict */ out
) {
    __m256d R0_0, R1_0,
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
        R0_5 = _mm256_load_pd(in + r * cols); // Aligned load
        R1_5 = _mm256_loadu_pd(in + r * cols + 1); // Unligned load
        R0_7 = (__m256d) _mm256_permute2x128_si256((__m256i) R0_5, (__m256i) R0_5, _MM_SHUFFLE(0, 0, 2, 0)); // Step 1 of bitshift
        R1_7 = (__m256d) _mm256_and_si256((__m256i) R0_5, (__m256i) R1_5);
        R0_7 = (__m256d) _mm256_alignr_epi8((__m256i) R0_5, (__m256i) R0_7, 16 - 1); // Step 2 if bitshift RIGHT
        R0_7 = (__m256d) _mm256_and_si256((__m256i) R0_7, (__m256i) R1_7);
        _mm256_store_pd(out + r * cols, R0_7);

        // Precompute pointers
        uint8_t* ptr0 = in + r * cols;
        uint8_t* ptr1 = in + r * cols;
        uint8_t* ptr2 = in + r * cols;
        uint8_t* ptr3 = in + r * cols;
        uint8_t* ptr4 = in + r * cols;
        uint8_t* ptr5 = in + r * cols;
        uint8_t* ptr6 = in + r * cols;
        uint8_t* ptr7 = in + r * cols;

        // For every SIMD load of row besides first and last
        for (int c = 1; c < ((cols/SIMD_N_ELEM)-1); c+= R_BATCHES) {
            // Load first 2 vals, perform and, load next val, perform final and then store
            #if R_BATCHES > 0
                uint8_t* cptr0 = ptr0 + (c + 0) * SIMD_N_ELEM;
            #endif
            #if R_BATCHES > 1
                uint8_t* cptr1 = ptr1 + (c + 1) * SIMD_N_ELEM;
            #endif
            #if R_BATCHES > 2
                uint8_t* cptr2 = ptr2 + (c + 2) * SIMD_N_ELEM;
            #endif
            #if R_BATCHES > 3
                uint8_t* cptr3 = ptr3 + (c + 3) * SIMD_N_ELEM;
            #endif
            #if R_BATCHES > 4
                uint8_t* cptr4 = ptr4 + (c + 4) * SIMD_N_ELEM;
            #endif
            #if R_BATCHES > 5
                uint8_t* cptr5 = ptr5 + (c + 5) * SIMD_N_ELEM;
            #endif
            #if R_BATCHES > 6
                uint8_t* cptr6 = ptr6 + (c + 6) * SIMD_N_ELEM;
            #endif
            #if R_BATCHES > 7
                uint8_t* cptr7 = ptr7 + (c + 7) * SIMD_N_ELEM;
            #endif

            // Itr 1
            #if R_BATCHES >= 1
            #ifndef DEBUG_UNALIGNED
            R0_0 = _mm256_loadu_pd(cptr0 - 1);
            #else
            R0_0 = _mm256_load_pd(cptr0);
            #endif
            R1_0 = _mm256_load_pd(cptr0);
            R0_0 = (__m256d) _mm256_and_si256((__m256i) R0_0, (__m256i) R1_0);
            #ifndef DEBUG_UNALIGNED
            R1_0 = _mm256_loadu_pd(cptr0 + 1);
            #else
            R1_0 = _mm256_load_pd(cptr0);
            #endif
            R0_0 = (__m256d) _mm256_and_si256((__m256i) R0_0, (__m256i) R1_0);
            _mm256_store_pd(out + r * cols + (c + 0) * SIMD_N_ELEM, R0_0);
            #endif

            // Itr 2
            #if R_BATCHES >= 2
            #ifndef DEBUG_UNALIGNED
            R0_1 = _mm256_loadu_pd(cptr1 - 1);
            #else
            R0_1 = _mm256_load_pd(cptr1);
            #endif
            R1_1 = _mm256_load_pd(cptr1);
            R0_1 = (__m256d) _mm256_and_si256((__m256i) R0_1, (__m256i) R1_1);
            #ifndef DEBUG_UNALIGNED
            R1_1 = _mm256_loadu_pd(cptr1 + 1);
            #else
            R1_1 = _mm256_load_pd(cptr1);
            #endif
            R0_1 = (__m256d) _mm256_and_si256((__m256i) R0_1, (__m256i) R1_1);
            _mm256_store_pd(out + r * cols + (c + 1) * SIMD_N_ELEM, R0_1);
            #endif

            // Itr 3
            #if R_BATCHES >= 3
            #ifndef DEBUG_UNALIGNED
            R0_2 = _mm256_loadu_pd(cptr2 - 1);
            #else
            R0_2 = _mm256_load_pd(cptr2);
            #endif
            R1_2 = _mm256_load_pd(cptr2);
            R0_2 = (__m256d) _mm256_and_si256((__m256i) R0_2, (__m256i) R1_2);
            #ifndef DEBUG_UNALIGNED
            R1_2 = _mm256_loadu_pd(cptr2 + 1);
            #else
            R1_2 = _mm256_load_pd(cptr2);
            #endif
            R0_2 = (__m256d) _mm256_and_si256((__m256i) R0_2, (__m256i) R1_2);
            _mm256_store_pd(out + r * cols + (c + 2) * SIMD_N_ELEM, R0_2);
            #endif

            // Itr 4
            #if R_BATCHES >= 4
            #ifndef DEBUG_UNALIGNED
            R0_3 = _mm256_loadu_pd(cptr3 - 1);
            #else
            R0_3 = _mm256_load_pd(cptr3);
            #endif
            R1_3 = _mm256_load_pd(cptr3);
            R0_3 = (__m256d) _mm256_and_si256((__m256i) R0_3, (__m256i) R1_3);
            #ifndef DEBUG_UNALIGNED
            R1_3 = _mm256_loadu_pd(cptr3 + 1);
            #else
            R1_3 = _mm256_load_pd(cptr3);
            #endif
            R0_3 = (__m256d) _mm256_and_si256((__m256i) R0_3, (__m256i) R1_3);
            _mm256_store_pd(out + r * cols + (c + 3) * SIMD_N_ELEM, R0_3);
            #endif

            // Itr 5
            #if R_BATCHES >= 5
            #ifndef DEBUG_UNALIGNED
            R0_4 = _mm256_loadu_pd(cptr4 - 1);
            #else
            R0_4 = _mm256_load_pd(cptr4);
            #endif
            R1_4 = _mm256_load_pd(cptr4);
            R0_4 = (__m256d) _mm256_and_si256((__m256i) R0_4, (__m256i) R1_4);
            #ifndef DEBUG_UNALIGNED
            R1_4 = _mm256_loadu_pd(cptr4 + 1);
            #else
            R1_4 = _mm256_load_pd(cptr4);
            #endif
            R0_4 = (__m256d) _mm256_and_si256((__m256i) R0_4, (__m256i) R1_4);
            _mm256_store_pd(out + r * cols + (c + 4) * SIMD_N_ELEM, R0_4);
            #endif

            // Itr 6
            #if R_BATCHES >= 6
            #ifndef DEBUG_UNALIGNED
            R0_5 = _mm256_loadu_pd(cptr5 - 1);
            #else
            R0_5 = _mm256_load_pd(cptr5);
            #endif
            R1_5 = _mm256_load_pd(cptr5);
            R0_5 = (__m256d) _mm256_and_si256((__m256i) R0_5, (__m256i) R1_5);
            #ifndef DEBUG_UNALIGNED
            R1_5 = _mm256_loadu_pd(cptr5 + 1);
            #else
            R1_5 = _mm256_load_pd(cptr5);
            #endif
            R0_5 = (__m256d) _mm256_and_si256((__m256i) R0_5, (__m256i) R1_5);
            _mm256_store_pd(out + r * cols + (c + 5) * SIMD_N_ELEM, R0_5);
            #endif

            // Itr 7
            #if R_BATCHES >= 7
            #ifndef DEBUG_UNALIGNED
            R0_6 = _mm256_loadu_pd(cptr6 - 1);
            #else
            R0_6 = _mm256_load_pd(cptr6);
            #endif
            R1_6 = _mm256_load_pd(cptr6);
            R0_6 = (__m256d) _mm256_and_si256((__m256i) R0_6, (__m256i) R1_6);
            #ifndef DEBUG_UNALIGNED
            R1_6 = _mm256_loadu_pd(cptr6 + 1);
            #else
            R1_6 = _mm256_load_pd(cptr6);
            #endif
            R0_6 = (__m256d) _mm256_and_si256((__m256i) R0_6, (__m256i) R1_6);
            _mm256_store_pd(out + r * cols + (c + 6) * SIMD_N_ELEM, R0_6);
            #endif

            // Itr 8
            #if R_BATCHES >= 8
            #ifndef DEBUG_UNALIGNED
            R0_7 = _mm256_loadu_pd(cptr7 - 1);
            #else
            R0_7 = _mm256_load_pd(cptr7);
            #endif
            R1_7 = _mm256_load_pd(cptr7);
            R0_7 = (__m256d) _mm256_and_si256((__m256i) R0_7, (__m256i) R1_7);
            #ifndef DEBUG_UNALIGNED
            R1_7 = _mm256_loadu_pd(cptr7 + 1);
            #else
            R1_7 = _mm256_load_pd(cptr7);
            #endif
            R0_7 = (__m256d) _mm256_and_si256((__m256i) R0_7, (__m256i) R1_7);
            _mm256_store_pd(out + r * cols + (c + 7) * SIMD_N_ELEM, R0_7);
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
        R0_0 = _mm256_load_pd(in + (r+1) * cols - SIMD_N_ELEM); // Aligned load
        R1_0 = _mm256_loadu_pd(in + (r+1) * cols - SIMD_N_ELEM - 1); // Unligned load
        R0_1 = (__m256d) _mm256_permute2x128_si256((__m256i) R0_0, (__m256i) R0_0, _MM_SHUFFLE(2, 0, 0, 1)); // Step 1 of bitshift
        R1_1 = (__m256d) _mm256_and_si256((__m256i) R0_0, (__m256i) R1_0);
        R0_1 = (__m256d) _mm256_alignr_epi8((__m256i) R0_1, (__m256i) R0_0, 1); // Step 2 if bitshift LEFFT
        R0_1 = (__m256d) _mm256_and_si256((__m256i) R0_1, (__m256i) R1_1);
        _mm256_store_pd(out + (r+1) * cols - SIMD_N_ELEM, R0_1);

    }
}

void rows_kernel_alt_2_2
(
    int cols,
    int rows,
    int8_t* /* restrict */ in,
    int8_t* /* restrict */ out
) {
    __m256d ymm0, ymm8,
            ymm1, ymm9,
            ymm2, ymm10,
            ymm3, ymm11,
            ymm4, ymm12,
            ymm5, ymm13,
            ymm6, ymm14,
            ymm7, ymm15;

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
        ymm5 = _mm256_load_pd(in + r * cols); // Aligned load
        ymm13 = _mm256_loadu_pd(in + r * cols + 1); // Unligned load
        ymm7 = (__m256d) _mm256_permute2x128_si256((__m256i) ymm5, (__m256i) ymm5, _MM_SHUFFLE(0, 0, 2, 0)); // Step 1 of bitshift
        ymm15 = (__m256d) _mm256_and_si256((__m256i) ymm5, (__m256i) ymm13);
        ymm7 = (__m256d) _mm256_alignr_epi8((__m256i) ymm5, (__m256i) ymm7, 16 - 1); // Step 2 if bitshift RIGHT
        ymm7 = (__m256d) _mm256_and_si256((__m256i) ymm7, (__m256i) ymm15);
        _mm256_store_pd(out + r * cols, ymm7);

        // Precompute pointers
        #if R_BATCHES > 0
        uint8_t* ptr0 = in + r * cols;
        #endif
        #if R_BATCHES > 1
        uint8_t* ptr1 = in + r * cols;
        #endif
        #if R_BATCHES > 2
        uint8_t* ptr2 = in + r * cols;
        #endif
        #if R_BATCHES > 3
        uint8_t* ptr3 = in + r * cols;
        #endif
        #if R_BATCHES > 4
        uint8_t* ptr4 = in + r * cols;
        #endif
        #if R_BATCHES > 5
        uint8_t* ptr5 = in + r * cols;
        #endif
        #if R_BATCHES > 6
        uint8_t* ptr6 = in + r * cols;
        #endif
        #if R_BATCHES > 7
        uint8_t* ptr7 = in + r * cols;
        #endif

        // For every SIMD load of row besides first and last
        for (int c = 1; c < ((cols/SIMD_N_ELEM)-1); c+= R_BATCHES) {
            // Load first 2 vals, perform and, load next val, perform final and then store
            #if R_BATCHES > 0
                uint8_t* cptr0 = ptr0 + (c + 0) * SIMD_N_ELEM;
            #endif
            #if R_BATCHES > 1
                uint8_t* cptr1 = ptr1 + (c + 1) * SIMD_N_ELEM;
            #endif
            #if R_BATCHES > 2
                uint8_t* cptr2 = ptr2 + (c + 2) * SIMD_N_ELEM;
            #endif
            #if R_BATCHES > 3
                uint8_t* cptr3 = ptr3 + (c + 3) * SIMD_N_ELEM;
            #endif
            #if R_BATCHES > 4
                uint8_t* cptr4 = ptr4 + (c + 4) * SIMD_N_ELEM;
            #endif
            #if R_BATCHES > 5
                uint8_t* cptr5 = ptr5 + (c + 5) * SIMD_N_ELEM;
            #endif
            #if R_BATCHES > 6
                uint8_t* cptr6 = ptr6 + (c + 6) * SIMD_N_ELEM;
            #endif
            #if R_BATCHES > 7
                uint8_t* cptr7 = ptr7 + (c + 7) * SIMD_N_ELEM;
            #endif

            // Itr 1
            #if R_BATCHES >= 1
            #ifndef DEBUG_UNALIGNED
            ymm0 = _mm256_loadu_pd(cptr0 - 1);
            #else
            ymm0 = _mm256_load_pd(cptr0);
            #endif
            ymm8 = _mm256_load_pd(cptr0);
            ymm0 = (__m256d) _mm256_and_si256((__m256i) ymm0, (__m256i) ymm8);
            #ifndef DEBUG_UNALIGNED
            ymm8 = _mm256_loadu_pd(cptr0 + 1);
            #else
            ymm8 = _mm256_load_pd(cptr0);
            #endif
            ymm0 = (__m256d) _mm256_and_si256((__m256i) ymm0, (__m256i) ymm8);
            _mm256_store_pd(out + r * cols + (c + 0) * SIMD_N_ELEM, ymm0);
            #endif

            // Itr 2
            #if R_BATCHES >= 2
            #ifndef DEBUG_UNALIGNED
            ymm1 = _mm256_loadu_pd(cptr1 - 1);
            #else
            ymm1 = _mm256_load_pd(cptr1);
            #endif
            ymm9 = _mm256_load_pd(cptr1);
            ymm1 = (__m256d) _mm256_and_si256((__m256i) ymm1, (__m256i) ymm9);
            #ifndef DEBUG_UNALIGNED
            ymm9 = _mm256_loadu_pd(cptr1 + 1);
            #else
            ymm9 = _mm256_load_pd(cptr1);
            #endif
            ymm1 = (__m256d) _mm256_and_si256((__m256i) ymm1, (__m256i) ymm9);
            _mm256_store_pd(out + r * cols + (c + 1) * SIMD_N_ELEM, ymm1);
            #endif

            // Itr 3
            #if R_BATCHES >= 3
            #ifndef DEBUG_UNALIGNED
            ymm2 = _mm256_loadu_pd(cptr2 - 1);
            #else
            ymm2 = _mm256_load_pd(cptr2);
            #endif
            ymm10 = _mm256_load_pd(cptr2);
            ymm2 = (__m256d) _mm256_and_si256((__m256i) ymm2, (__m256i) ymm10);
            #ifndef DEBUG_UNALIGNED
            ymm10 = _mm256_loadu_pd(cptr2 + 1);
            #else
            ymm10 = _mm256_load_pd(cptr2);
            #endif
            ymm2 = (__m256d) _mm256_and_si256((__m256i) ymm2, (__m256i) ymm10);
            _mm256_store_pd(out + r * cols + (c + 2) * SIMD_N_ELEM, ymm2);
            #endif

            // Itr 4
            #if R_BATCHES >= 4
            #ifndef DEBUG_UNALIGNED
            ymm3 = _mm256_loadu_pd(cptr3 - 1);
            #else
            ymm3 = _mm256_load_pd(cptr3);
            #endif
            ymm11 = _mm256_load_pd(cptr3);
            ymm3 = (__m256d) _mm256_and_si256((__m256i) ymm3, (__m256i) ymm11);
            #ifndef DEBUG_UNALIGNED
            ymm11 = _mm256_loadu_pd(cptr3 + 1);
            #else
            ymm11 = _mm256_load_pd(cptr3);
            #endif
            ymm3 = (__m256d) _mm256_and_si256((__m256i) ymm3, (__m256i) ymm11);
            _mm256_store_pd(out + r * cols + (c + 3) * SIMD_N_ELEM, ymm3);
            #endif

            // Itr 5
            #if R_BATCHES >= 5
            #ifndef DEBUG_UNALIGNED
            ymm4 = _mm256_loadu_pd(cptr4 - 1);
            #else
            ymm4 = _mm256_load_pd(cptr4);
            #endif
            ymm12 = _mm256_load_pd(cptr4);
            ymm4 = (__m256d) _mm256_and_si256((__m256i) ymm4, (__m256i) ymm12);
            #ifndef DEBUG_UNALIGNED
            ymm12 = _mm256_loadu_pd(cptr4 + 1);
            #else
            ymm12 = _mm256_load_pd(cptr4);
            #endif
            ymm4 = (__m256d) _mm256_and_si256((__m256i) ymm4, (__m256i) ymm12);
            _mm256_store_pd(out + r * cols + (c + 4) * SIMD_N_ELEM, ymm4);
            #endif

            // Itr 6
            #if R_BATCHES >= 6
            #ifndef DEBUG_UNALIGNED
            ymm5 = _mm256_loadu_pd(cptr5 - 1);
            #else
            ymm5 = _mm256_load_pd(cptr5);
            #endif
            ymm13 = _mm256_load_pd(cptr5);
            ymm5 = (__m256d) _mm256_and_si256((__m256i) ymm5, (__m256i) ymm13);
            #ifndef DEBUG_UNALIGNED
            ymm13 = _mm256_loadu_pd(cptr5 + 1);
            #else
            ymm13 = _mm256_load_pd(cptr5);
            #endif
            ymm5 = (__m256d) _mm256_and_si256((__m256i) ymm5, (__m256i) ymm13);
            _mm256_store_pd(out + r * cols + (c + 5) * SIMD_N_ELEM, ymm5);
            #endif

            // Itr 7
            #if R_BATCHES >= 7
            #ifndef DEBUG_UNALIGNED
            ymm6 = _mm256_loadu_pd(cptr6 - 1);
            #else
            ymm6 = _mm256_load_pd(cptr6);
            #endif
            ymm14 = _mm256_load_pd(cptr6);
            ymm6 = (__m256d) _mm256_and_si256((__m256i) ymm6, (__m256i) ymm14);
            #ifndef DEBUG_UNALIGNED
            ymm14 = _mm256_loadu_pd(cptr6 + 1);
            #else
            ymm14 = _mm256_load_pd(cptr6);
            #endif
            ymm6 = (__m256d) _mm256_and_si256((__m256i) ymm6, (__m256i) ymm14);
            _mm256_store_pd(out + r * cols + (c + 6) * SIMD_N_ELEM, ymm6);
            #endif

            // Itr 8
            #if R_BATCHES >= 8
            #ifndef DEBUG_UNALIGNED
            ymm7 = _mm256_loadu_pd(cptr7 - 1);
            #else
            ymm7 = _mm256_load_pd(cptr7);
            #endif
            ymm15 = _mm256_load_pd(cptr7);
            ymm7 = (__m256d) _mm256_and_si256((__m256i) ymm7, (__m256i) ymm15);
            #ifndef DEBUG_UNALIGNED
            ymm15 = _mm256_loadu_pd(cptr7 + 1);
            #else
            ymm15 = _mm256_load_pd(cptr7);
            #endif
            ymm7 = (__m256d) _mm256_and_si256((__m256i) ymm7, (__m256i) ymm15);
            _mm256_store_pd(out + r * cols + (c + 7) * SIMD_N_ELEM, ymm7);
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
        ymm8 = _mm256_loadu_pd(in + (r+1) * cols - SIMD_N_ELEM - 1); // Unligned load
        ymm1 = (__m256d) _mm256_permute2x128_si256((__m256i) ymm0, (__m256i) ymm0, _MM_SHUFFLE(2, 0, 0, 1)); // Step 1 of bitshift
        ymm9 = (__m256d) _mm256_and_si256((__m256i) ymm0, (__m256i) ymm8);
        ymm1 = (__m256d) _mm256_alignr_epi8((__m256i) ymm1, (__m256i) ymm0, 9); // Step 2 if bitshift LEFFT
        ymm1 = (__m256d) _mm256_and_si256((__m256i) ymm1, (__m256i) ymm1);
        _mm256_store_pd(out + (r+1) * cols - SIMD_N_ELEM, ymm1);

    }
}
