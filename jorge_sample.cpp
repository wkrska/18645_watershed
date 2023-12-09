#include <algorithm>
#include <cmath>
#include <cstdio>
#include <immintrin.h>
#include <iostream>
#include <iomanip> 
#include <opencv2/opencv.hpp>

#define MAX_FREQ 3.2
#define BASE_FREQ 2.4
#define RUNS 100000

//#define A 1.0 //horizantal distance weight
//#define B 1.4 //diagonal distance weight
//#define C 2.1969 //knights move distance weight
#define WEIGHT_SCALE 7.0

#define A WEIGHT_SCALE*1.0 //horizantal distance weight
#define B WEIGHT_SCALE*1.4 //diagonal distance weight
#define C WEIGHT_SCALE*2.1969 //knights move distance weight

#define NUM_ROWS 316
#define NUM_COLS 256
/*
5x5 Eucledian Distance Transform
*/

static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

//Forward Pass Kernel
void fp_kernel(double *dst){
    __m256d A_VEC = _mm256_set1_pd(A);
    __m256d A_VEC2 = _mm256_set1_pd(A*2.0);
    __m256d B_VEC = _mm256_set1_pd(B);
    __m256d B_VEC2 = _mm256_set1_pd(B*2.0);
    __m256d C_VEC = _mm256_set1_pd(C);

    //top top left rows
    double *top_left = &dst[0 - (NUM_COLS*2) - 2];

    __m256d tl0 = _mm256_loadu_pd(&top_left[0]);
    __m256d tl1 = _mm256_loadu_pd(&top_left[1]);
    __m256d tl2 = _mm256_loadu_pd(&top_left[2]);
    __m256d tl3 = _mm256_loadu_pd(&top_left[3]);
    __m256d tl4 = _mm256_loadu_pd(&top_left[4]);

    __m256d temp_r0 = _mm256_add_pd(tl0, B_VEC2);
    __m256d temp_r1 = _mm256_add_pd(tl1, C_VEC);
    __m256d temp_r2 = _mm256_add_pd(tl2, A_VEC2);
    __m256d temp_r3 = _mm256_add_pd(tl3, C_VEC);
    __m256d temp_r4 = _mm256_add_pd(tl4, B_VEC2);

    __m256d tl_res = _mm256_min_pd(temp_r0, temp_r1);
    tl_res = _mm256_min_pd(tl_res, temp_r2);
    tl_res = _mm256_min_pd(tl_res, temp_r3);
    tl_res = _mm256_min_pd(tl_res, temp_r4);

    //top left rows
    top_left = &dst[0 - (NUM_COLS) - 2];

    tl0 = _mm256_loadu_pd(&top_left[0]);
    tl1 = _mm256_loadu_pd(&top_left[1]);
    tl2 = _mm256_loadu_pd(&top_left[2]);
    tl3 = _mm256_loadu_pd(&top_left[3]);
    tl4 = _mm256_loadu_pd(&top_left[4]);

    temp_r0 = _mm256_add_pd(tl0, C_VEC);
    temp_r1 = _mm256_add_pd(tl1, B_VEC);
    temp_r2 = _mm256_add_pd(tl2, A_VEC);
    temp_r3 = _mm256_add_pd(tl3, B_VEC);
    temp_r4 = _mm256_add_pd(tl4, C_VEC);

    tl_res = _mm256_min_pd(tl_res, temp_r0);
    tl_res = _mm256_min_pd(tl_res, temp_r1);
    tl_res = _mm256_min_pd(tl_res, temp_r2);
    tl_res = _mm256_min_pd(tl_res, temp_r3);
    tl_res = _mm256_min_pd(tl_res, temp_r4);

    __m256d res = _mm256_load_pd(&dst[0]);
    res = _mm256_min_pd(res, tl_res);

    _mm256_store_pd(&dst[0], res);

    //values to the left:
    //This part is unavoidably sequential
    //It must be done at the END and cannot be done concurrently
    double left_tmp = std::min(dst[0-1] + A, dst[0-2] + 2*A);
    dst[0] = std::min(dst[0], left_tmp);
    left_tmp = std::min(dst[1-1] + A, dst[1-2] + 2*A);
    dst[1] = std::min(dst[1], left_tmp);
    left_tmp = std::min(dst[2-1] + A, dst[2-2] + 2*A);
    dst[2] = std::min(dst[2], left_tmp);
    left_tmp = std::min(dst[3-1] + A, dst[3-2] + 2*A);
    dst[3] = std::min(dst[3], left_tmp);
}   

//Backward Pass Kernel
void bp_kernel(double *dst){

    __m256d A_VEC = _mm256_set1_pd(A);
    __m256d A_VEC2 = _mm256_set1_pd(A*2.0);
    __m256d B_VEC = _mm256_set1_pd(B);
    __m256d B_VEC2 = _mm256_set1_pd(B*2.0);
    __m256d C_VEC = _mm256_set1_pd(C);

    //bottom bottom left rows
    double *bottom_left = &dst[0 + (NUM_COLS*2) - 2];

    __m256d tl0 = _mm256_loadu_pd(&bottom_left[0]);
    __m256d tl1 = _mm256_loadu_pd(&bottom_left[1]);
    __m256d tl2 = _mm256_loadu_pd(&bottom_left[2]);
    __m256d tl3 = _mm256_loadu_pd(&bottom_left[3]);
    __m256d tl4 = _mm256_loadu_pd(&bottom_left[4]);

    __m256d temp_r0 = _mm256_add_pd(tl0, B_VEC2);
    __m256d temp_r1 = _mm256_add_pd(tl1, C_VEC);
    __m256d temp_r2 = _mm256_add_pd(tl2, A_VEC2);
    __m256d temp_r3 = _mm256_add_pd(tl3, C_VEC);
    __m256d temp_r4 = _mm256_add_pd(tl4, B_VEC2);

    __m256d tl_res = _mm256_min_pd(temp_r0, temp_r1);
    tl_res = _mm256_min_pd(tl_res, temp_r2);
    tl_res = _mm256_min_pd(tl_res, temp_r3);
    tl_res = _mm256_min_pd(tl_res, temp_r4);

    //bottom left rows
    bottom_left = &dst[0 + (NUM_COLS) - 2];

    tl0 = _mm256_loadu_pd(&bottom_left[0]);
    tl1 = _mm256_loadu_pd(&bottom_left[1]);
    tl2 = _mm256_loadu_pd(&bottom_left[2]);
    tl3 = _mm256_loadu_pd(&bottom_left[3]);
    tl4 = _mm256_loadu_pd(&bottom_left[4]);

    temp_r0 = _mm256_add_pd(tl0, C_VEC);
    temp_r1 = _mm256_add_pd(tl1, B_VEC);
    temp_r2 = _mm256_add_pd(tl2, A_VEC);
    temp_r3 = _mm256_add_pd(tl3, B_VEC);
    temp_r4 = _mm256_add_pd(tl4, C_VEC);

    tl_res = _mm256_min_pd(tl_res, temp_r0);
    tl_res = _mm256_min_pd(tl_res, temp_r1);
    tl_res = _mm256_min_pd(tl_res, temp_r2);
    tl_res = _mm256_min_pd(tl_res, temp_r3);
    tl_res = _mm256_min_pd(tl_res, temp_r4);

    __m256d res = _mm256_load_pd(&dst[0]);
    res = _mm256_min_pd(res, tl_res);

    _mm256_store_pd(&dst[0], res);

    //values to the right:
    //This part is unavoidably sequential
    //It must be done at the END and cannot be done concurrently
    double right_tmp = std::min(dst[3+1] + A, dst[3+2] + 2*A);
    dst[3] = std::min(dst[3], right_tmp);
    right_tmp = std::min(dst[2+1] + A, dst[2+2] + 2*A);
    dst[2] = std::min(dst[2], right_tmp);
    right_tmp = std::min(dst[1+1] + A, dst[1+2] + 2*A);
    dst[1] = std::min(dst[1], right_tmp);
    right_tmp = std::min(dst[0+1] + A, dst[0+2] + 2*A);
    dst[0] = std::min(dst[0], right_tmp);
}   

cv::Mat convertDoubleArrayToMat(const double* data, int rows, int cols) {
    // Assuming the input data is in row-major order
    cv::Mat mat(rows, cols, CV_64F);
    memcpy(mat.data, data, rows * cols * sizeof(double));
    return mat;
}

int main() {

    cv::Mat originalImage = cv::imread("binary_image.jpg", cv::IMREAD_GRAYSCALE);

    if (originalImage.empty()) {
        std::cerr << "Error loading the image." << std::endl;
        return -1;
    }

    // Define padding size
    int padSize = 2;

    // Create a new padded image
    cv::Mat paddedImage(originalImage.rows + 2 * padSize, originalImage.cols + 2 * padSize, CV_8U, cv::Scalar(0));

    // Copy the original image to the center of the padded image
    originalImage.copyTo(paddedImage(cv::Rect(padSize, padSize, originalImage.cols, originalImage.rows)));

    // Convert the padded image to a double*
    double* imageData = new double[paddedImage.rows * paddedImage.cols];

    for (int i = 0; i < paddedImage.rows; ++i) {
        for (int j = 0; j < paddedImage.cols; ++j) {
            imageData[i * paddedImage.cols + j] = static_cast<double>(paddedImage.at<uchar>(i, j));
        }
    }

    //printf("dims %d %d",paddedImage.rows,paddedImage.cols);
    for(int i = 2; i < NUM_ROWS-2; i=i+1) {
        for(int j = 2; j < NUM_COLS-2; j=j+4) {
            //printf("i: %d j: %d addr: %d\n",i,j,i*NUM_COLS + j);
            fp_kernel(&imageData[i*NUM_COLS + j]);
        }
    }
    for (int i = (NUM_ROWS-2); i >= 2; i -= 1) {
        for (int j = (NUM_COLS - 2 - 4); j >= 2; j -= 4) {
            bp_kernel(&imageData[i * NUM_COLS + j]);
        }
    }

    // Now imageData is a double* representing the pixel values of the image
    cv::Mat imageMat = convertDoubleArrayToMat(imageData, NUM_ROWS, NUM_COLS);

    // Save the Mat object as a JPEG file
    std::string filename = "res.jpg";
    bool success = cv::imwrite(filename, imageMat);

    if (success) {
        std::cout << "Image saved successfully as " << filename << std::endl;
    } else {
        std::cerr << "Error saving image" << std::endl;
    }
    
    delete[] imageData;



    return 0;
}