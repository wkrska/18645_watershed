#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
#include <immintrin.h>

//TODO: Adjust the frequency based on your machine.
#define MAX_FREQ 3.4
#define BASE_FREQ 2.4

//TODO: Change this to reflect the number of instructions in your chain
#define NUM_INST 1000.0

static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}


// TODO: Define your macros here
#define SIMD_AND10(a, b) \
  a = _mm256_and_si256(b, a); \
  a = _mm256_and_si256(b, a); \
  a = _mm256_and_si256(b, a); \
  a = _mm256_and_si256(b, a); \
  a = _mm256_and_si256(b, a); \
  a = _mm256_and_si256(b, a); \
  a = _mm256_and_si256(b, a); \
  a = _mm256_and_si256(b, a); \
  a = _mm256_and_si256(b, a); \
  a = _mm256_and_si256(b, a);

#define SIMD_AND100(a, b) \
  SIMD_AND10(a, b); \
  SIMD_AND10(a, b); \
  SIMD_AND10(a, b); \
  SIMD_AND10(a, b); \
  SIMD_AND10(a, b); \
  SIMD_AND10(a, b); \
  SIMD_AND10(a, b); \
  SIMD_AND10(a, b); \
  SIMD_AND10(a, b); \
  SIMD_AND10(a, b);

#define SIMD_AND1000(a, b) \
  SIMD_AND100(a, b); \
  SIMD_AND100(a, b); \
  SIMD_AND100(a, b); \
  SIMD_AND100(a, b); \
  SIMD_AND100(a, b); \
  SIMD_AND100(a, b); \
  SIMD_AND100(a, b); \
  SIMD_AND100(a, b); \
  SIMD_AND100(a, b); \
  SIMD_AND100(a, b);
//Hint: You may want to write Macros that call Macro intrinsics

int main(int argc, char **argv) {

  // int runs = atoi(argv[1]);
  // You might want to use the above code to control number of runs.
  int runs = 100;
  

  unsigned long long st;
  unsigned long long et;
  unsigned long long sum = 0;

  long long int *v0, *v1, *out;
  posix_memalign((void**) &v0, 64, 8*sizeof(int));
  posix_memalign((void**) &v1, 64, 8*sizeof(int));
  posix_memalign((void**) &out, 64, 8*sizeof(int));

  for (int i = 0; i < 8; i++) {
    v0[i] = i*0xf;
    v1[i] = i*0xe;
  }

  volatile __m256i a, b;
  a = _mm256_set_epi32(v0[0],
                         v0[1],
                         v0[2],
                         v0[3],
                         v0[4],
                         v0[5],
                         v0[6],
                         v0[7]);
    b = _mm256_set_epi32(v1[0],
                         v1[1],
                         v1[2],
                         v1[3],
                         v1[4],
                         v1[5],
                         v1[6],
                         v1[7]);
  for (int j = 0; j < runs; j++) {

    // Start timing
    st = rdtsc();

    // Run 1000 itr
    SIMD_AND1000(a,b);

    // Stop timing
    et = rdtsc();
    
    // Calc runtime
    sum += et-st;
  }
  // Retrieve values to prevent from being optimized away
    _mm256_store_si256((__m256i*) out,a);
    for (int k = 0; k < 4; k++)
        printf("%lld ", out[k]);

  printf("RDTSC Base Cycles Taken for SIMD_AND: %llu\n\r",sum);
  printf("Latency: %lf\n\r", MAX_FREQ/BASE_FREQ * sum / (NUM_INST*runs));

  return 0;
}
