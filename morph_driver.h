/* usage:
morph_driver(mat_in, mat_out, op_code, offset_size, iterations)
*/
#include <opencv2/core.hpp> // to be able to use Mat class
#include <stdlib.h>
#include <functional>
#include "immintrin.h"

#define R_STEP (6)

using namespace std;
using namespace cv;

template <class T>
class myMat {
    private:
        unsigned int rows; 
        unsigned int cols;
        unsigned int data_size;

    public:
        T *image; 
        // Constructor
        myMat(unsigned int rows = 0, unsigned int cols = 0) : rows(rows), cols(cols) {
            data_size = sizeof(T);
            posix_memalign((void**) &image, 64, rows * cols * data_size);
        }

        int getRows() { return rows; }
        
        int getCols() { return cols; }

        int getDS() { return data_size; }
};

template <class T>
void morph_kernel(myMat<T> &mat_in, myMat<T> &mat_out) {
    // Number of columns / 256 bit reads --> Number of columns / (cols / 256 bits) = num cols / (256/DS)
    __m256i I0, I1, I2, I3, I4, I5; // new inputs
    __m256i O0, O1, O2, O3, O4, O5; // outputs
    __m256i C0, C1, C2;

    for (int col = 0; col < mat_in.getCols(); col += (256/mat_in.getDS())) {
        for (int row = 0; row < mat_in.getRows(); row += R_STEP) {
            
        }
    }
}

template <class T>
void morph_driver(myMat<T> &mat_in, myMat<T> &mat_out, int op_code, int offset, int iteratons) {
    /* op_codes
        0 - erode
        1 - dilate
        2 - open
        3 - close
    */
    
    // Create offset size based on input op_code
    int true_offset;
    if (op_code == 2 || op_code == 3) {
        true_offset = offset + iterations;
    } else {
        true_offset = offset;
    }

    // Run kernel depending on op_code
    myMat<T> mat_temp(mat_in.getRows(), mat_in.getCols());
    switch (op_code) {
        case 0: // erode
            morph_kernel(mat_in, mat_out, true_offset,   [](__m256i &out, __m256i &a, __m256i &b){out = _mm256_and_si256(a, b)});
            transpose(mat_out, mat_temp);
            morph_kernel(mat_temp, mat_out, true_offset, [](__m256i &out, __m256i &a, __m256i &b){out = _mm256_and_si256(a, b)});
            break;
        case 1: // dilate
            morph_kernel(mat_in, mat_out, true_offset,   [](__m256i &out, __m256i &a, __m256i &b){out = _mm256_or_si256(a, b)});
            transpose(mat_out, mat_temp);
            morph_kernel(mat_temp, mat_out, true_offset, [](__m256i &out, __m256i &a, __m256i &b){out = _mm256_or_si256(a, b)});
            break;
        case 2: // open: erode then dilate
            morph_kernel(mat_in, mat_out, true_offset,   [](__m256i &out, __m256i &a, __m256i &b){out = _mm256_and_si256(a, b)});
            morph_kernel(mat_out, mat_temp, true_offset, [](__m256i &out, __m256i &a, __m256i &b){out = _mm256_or_si256(a, b)});
            transpose(mat_temp, mat_out);
            morph_kernel(mat_out, mat_temp, true_offset, [](__m256i &out, __m256i &a, __m256i &b){out = _mm256_and_si256(a, b)});
            morph_kernel(mat_temp, mat_out, true_offset, [](__m256i &out, __m256i &a, __m256i &b){out = _mm256_or_si256(a, b)});
            break;
        case 3: // close: dilate then erode
            morph_kernel(mat_in, mat_out, true_offset,   [](__m256i &out, __m256i &a, __m256i &b){out = _mm256_or_si256(a, b)});
            morph_kernel(mat_out, mat_temp, true_offset, [](__m256i &out, __m256i &a, __m256i &b){out = _mm256_and_si256(a, b)});
            transpose(mat_temp, mat_out);
            morph_kernel(mat_out, mat_temp, true_offset, [](__m256i &out, __m256i &a, __m256i &b){out = _mm256_or_si256(a, b)});
            morph_kernel(mat_temp, mat_out, true_offset, [](__m256i &out, __m256i &a, __m256i &b){out = _mm256_and_si256(a, b)});
            break;
        default:
            std::cout << "Invalid Opcode, range 0-3" << std::endl;
            return;
    }
}