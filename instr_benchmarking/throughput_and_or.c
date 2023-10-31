#include <stdio.h>
#include <string.h>
#include <x86intrin.h>
#include <immintrin.h>

#include <stdlib.h>

#define MAX_FREQ 3.4
#define BASE_FREQ 2.4
#define NUM_INST 100.0
#define NUM_CHAINS 10

static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

#define INIT_REG16_ARR(arr, a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a0A, a0B, a0C, a0D, a0E, a0F) \
  a00 = _mm256_loadu_pd(arr + (0x0*4)); \
  a01 = _mm256_loadu_pd(arr + (0x1*4)); \
  a02 = _mm256_loadu_pd(arr + (0x2*4)); \
  a03 = _mm256_loadu_pd(arr + (0x3*4)); \
  a04 = _mm256_loadu_pd(arr + (0x4*4)); \
  a05 = _mm256_loadu_pd(arr + (0x5*4)); \
  a06 = _mm256_loadu_pd(arr + (0x6*4)); \
  a07 = _mm256_loadu_pd(arr + (0x7*4)); \
  a08 = _mm256_loadu_pd(arr + (0x8*4)); \
  a09 = _mm256_loadu_pd(arr + (0x9*4)); \
  a0A = _mm256_loadu_pd(arr + (0xA*4)); \
  a0B = _mm256_loadu_pd(arr + (0xB*4)); \
  a0C = _mm256_loadu_pd(arr + (0xC*4)); \
  a0D = _mm256_loadu_pd(arr + (0xD*4)); \
  a0E = _mm256_loadu_pd(arr + (0xE*4)); \
  a0F = _mm256_loadu_pd(arr + (0xF*4));

#define INIT_REG16_ZERO(a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a0A, a0B, a0C, a0D, a0E, a0F) \
  a00 = _mm256_setzero_pd(); \
  a01 = _mm256_setzero_pd(); \
  a02 = _mm256_setzero_pd(); \
  a03 = _mm256_setzero_pd(); \
  a04 = _mm256_setzero_pd(); \
  a05 = _mm256_setzero_pd(); \
  a06 = _mm256_setzero_pd(); \
  a07 = _mm256_setzero_pd(); \
  a08 = _mm256_setzero_pd(); \
  a09 = _mm256_setzero_pd(); \
  a0A = _mm256_setzero_pd(); \
  a0B = _mm256_setzero_pd(); \
  a0C = _mm256_setzero_pd(); \
  a0D = _mm256_setzero_pd(); \
  a0E = _mm256_setzero_pd(); \
  a0F = _mm256_setzero_pd();

#define READ_REG16(arr, a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a0A, a0B, a0C, a0D, a0E, a0F) \
  _mm256_storeu_pd(arr + (0x0*4), a00); \
  _mm256_storeu_pd(arr + (0x1*4), a01); \
  _mm256_storeu_pd(arr + (0x2*4), a02); \
  _mm256_storeu_pd(arr + (0x3*4), a03); \
  _mm256_storeu_pd(arr + (0x4*4), a04); \
  _mm256_storeu_pd(arr + (0x5*4), a05); \
  _mm256_storeu_pd(arr + (0x6*4), a06); \
  _mm256_storeu_pd(arr + (0x7*4), a07); \
  _mm256_storeu_pd(arr + (0x8*4), a08); \
  _mm256_storeu_pd(arr + (0x9*4), a09); \
  _mm256_storeu_pd(arr + (0xA*4), a0A); \
  _mm256_storeu_pd(arr + (0xB*4), a0B); \
  _mm256_storeu_pd(arr + (0xC*4), a0C); \
  _mm256_storeu_pd(arr + (0xD*4), a0D); \
  _mm256_storeu_pd(arr + (0xE*4), a0E); \
  _mm256_storeu_pd(arr + (0xF*4), a0F); 

#define SIMD_FMA_NP(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F) \
  c00 = _mm256_fmadd_pd(a, b, c00); \
  c01 = _mm256_fmadd_pd(a, b, c01); \
  c02 = _mm256_fmadd_pd(a, b, c02); \
  c03 = _mm256_fmadd_pd(a, b, c03); \
  c04 = _mm256_fmadd_pd(a, b, c04); \
  c05 = _mm256_fmadd_pd(a, b, c05); \
  c06 = _mm256_fmadd_pd(a, b, c06); \
  c07 = _mm256_fmadd_pd(a, b, c07); \
  c08 = _mm256_fmadd_pd(a, b, c08); \
  c09 = _mm256_fmadd_pd(a, b, c09); /*\
  c0A = _mm256_fmadd_pd(a, b, c0A); /*\
  c0B = _mm256_fmadd_pd(a, b, c0B); /*\
  c0C = _mm256_fmadd_pd(a, b, c0C); /*\
  c0D = _mm256_fmadd_pd(a, b, c0D); /*\
  c0E = _mm256_fmadd_pd(a, b, c0E); /*\
  c0F = _mm256_fmadd_pd(a, b, c0F); /*\
  c10 = _mm256_fmadd_pd(a, b, c10); /*\
  c11 = _mm256_fmadd_pd(a, b, c11); /*\
  c12 = _mm256_fmadd_pd(a, b, c12); /*\
  c13 = _mm256_fmadd_pd(a, b, c13); /*\
  c14 = _mm256_fmadd_pd(a, b, c14); /*\
  c15 = _mm256_fmadd_pd(a, b, c15); /*\
  c16 = _mm256_fmadd_pd(a, b, c16); /*\
  c17 = _mm256_fmadd_pd(a, b, c17); /*\
  c18 = _mm256_fmadd_pd(a, b, c18); /*\
  c19 = _mm256_fmadd_pd(a, b, c19); /*\
  c1A = _mm256_fmadd_pd(a, b, c1A); /*\
  c1B = _mm256_fmadd_pd(a, b, c1B); /*\
  c1C = _mm256_fmadd_pd(a, b, c1C); /*\
  c1D = _mm256_fmadd_pd(a, b, c1D); /*\
  c1E = _mm256_fmadd_pd(a, b, c1E); /*\
  c1F = _mm256_fmadd_pd(a, b, c1F);
*/
#define SIMD_FMA10S(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F) \
  SIMD_FMA_NP(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F); \
  SIMD_FMA_NP(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F); \
  SIMD_FMA_NP(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F); \
  SIMD_FMA_NP(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F); \
  SIMD_FMA_NP(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F); \
  SIMD_FMA_NP(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F); \
  SIMD_FMA_NP(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F); \
  SIMD_FMA_NP(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F); \
  SIMD_FMA_NP(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F); \
  SIMD_FMA_NP(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F);

#define SIMD_FMA100S(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F) \
  SIMD_FMA10S(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F); \
  SIMD_FMA10S(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F); \
  SIMD_FMA10S(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F); \
  SIMD_FMA10S(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F); \
  SIMD_FMA10S(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F); \
  SIMD_FMA10S(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F); \
  SIMD_FMA10S(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F); \
  SIMD_FMA10S(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F); \
  SIMD_FMA10S(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F); \
  SIMD_FMA10S(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F);


int main(int argc, char **argv) {


  // int runs = atoi(argv[1]);
  // You might want to use the above code to control number of runs.
  int runs = 100;

  unsigned long long st;
  unsigned long long et;
  unsigned long long sum = 0;

  srand(0);

  for (int j = 0; j < runs; j++) {
    // Init registers with values
    double arr0[64], arr1[64], arr2[64];
    for (int i = 0; i<64; i++) {
      arr0[i] = (double) rand();
      arr1[i] = (double) rand();
      arr2[i] = (double) rand();
    }
    __m256d a,
            b,
            c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F;
    
    INIT_REG16_ARR(arr0, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F);
    INIT_REG16_ARR(arr1, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F);
    a = _mm256_loadu_pd(arr2);
    b = _mm256_loadu_pd(arr2 + 4);
    #pragma optimize("", off) // Got this off some random Intel docs, hopefully prevent things being optimized away

    // Start Timing
    st = rdtsc();

    // Run 1000 itr of up to 32 parallel chains (To change, change number of lines commented in SIMD_FMA_NP)
    SIMD_FMA100S(a, b, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F);

    // End time
    et = rdtsc();

    // Retrieve values to prevent from being optimized away
    double tmp_arr0[4*16], tmp_arr1[4*16];
    READ_REG16(tmp_arr0, c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c0E, c0F);
    READ_REG16(tmp_arr1, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, c1E, c1F);
    
    // Calc runtime
    sum += (et-st);

  }

  printf("RDTSC Base Cycles Taken for SIMD_FMA: %llu\n\r",sum);
  printf("TURBO Cycles Taken for SIMD_FMA: %lf\n\r",sum * ((double)MAX_FREQ)/BASE_FREQ);
  printf("Throughput : %lf (chains: %.0f)\n\r",((double)NUM_INST * runs * NUM_CHAINS) / (sum * MAX_FREQ/BASE_FREQ), (double) NUM_CHAINS);

return 0;
}
