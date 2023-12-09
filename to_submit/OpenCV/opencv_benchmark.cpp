#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "immintrin.h"

using namespace std;
using namespace cv;

#ifndef ROWS
#define ROWS 64
#endif
#ifndef COLS 
#define COLS *SIMD_N_ELEM
#endif

#ifndef RUNS
#define RUNS 1
#endif

#define MAX_FREQ 3.4
#define BASE_FREQ 2.4

#define DEBUG 1
// #define PRINTMAT

//timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

cv::Mat convertDoubleArrayToMat(const uint8_t* data, int rows, int cols) {
    // Assuming the input data is in row-major order
    cv::Mat mat(rows, cols, CV_8F);
    memcpy(mat.data, data, rows * cols * sizeof(uint8_t));
    return mat;
}

int main(int argc, char *argv[]) {
  // Image Generation : Generate an image of the same dimensions to directly compare with my impl (defined at top)
  // Make array 
  int8_t *in;
  posix_memalign((void**) &in, 64, ROWS * COLS * sizeof(int8_t));

  // Fill in with data
  for (int i = 0; i != ROWS*COLS; ++i)  { 
    #if (DEBUG == 1)
    int tile_size = 32;
    // Checker board with tiles of above size
    in[i] = (((i/(COLS*tile_size)) % 2 == 0) ^ (((i%(COLS))/tile_size) % 2 == 0)) ? 0 : ~0;
  }

  // Convert to MAT

  // Show the source image
  imshow("Source Image", src);
  printf("Source Image\n");
}