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

// This is NOT correct on your local machine
#define MAX_FREQ 3.4
#define BASE_FREQ 2.4

unsigned long long t0, t1;

//timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

int main(int argc, char *argv[])
{
    // Load the image
    CommandLineParser parser( argc, argv, "{@input | water_coins.jpg | input image}" );
    Mat src = imread( samples::findFile( parser.get<String>( "@input" ) ) );
    if( src.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }

    // Show the source image
    imshow("Source Image", src);
    printf("Source Image\n");

    // Create binary image from source image
    Mat bw;
    cvtColor(src, bw, COLOR_BGR2GRAY);
    // cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    t0 = rdtsc();
    threshold(bw, bw, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    // threshold(bw, bw, 250, 255, THRESH_BINARY_INV);
    t1 = rdtsc();
    imshow("Binary Image", bw);
    printf("Binary Image, Otse time = %llu\n",(t1-t0));

    // Noise removal on BW image
    Mat opening;
    Mat kernel3x3 = Mat::ones(3, 3, CV_8U);
    t0 = rdtsc();
    morphologyEx(bw, opening, 2, kernel3x3, Point(-1,-1), 2); // 2 is code for opening, see docs
    // morphologyEx(bw, opening, 3, kernel3x3, Point(-1,-1), 5); // 3 is code for close, see docs
    t1 = rdtsc();
    imshow("Opening", opening);
    printf("Opening Image, 2x dilate/erode time = %llu\n",(t1-t0));

    // Sure background
    Mat sure_bg;
    t0 = rdtsc();
    dilate(opening, sure_bg, kernel3x3, Point(-1,-1), 3);
    t1 = rdtsc();
    imshow("Sure BG", sure_bg);
    printf("BG Image, dilate time = %llu\n",(t1-t0));

    // Find sure foreground
    // Perform the distance transform algorithm
    Mat dist_transform;
    Mat sure_fg;
    t0 = rdtsc();
    distanceTransform(opening, dist_transform, DIST_L2, 5);
    t1 = rdtsc();
    double min, max;
    minMaxLoc(dist_transform, &min, &max);    
    threshold(dist_transform, sure_fg, 0.7*max, 255, THRESH_BINARY);
    imshow("Distance Transform Image", dist_transform/max);
    printf("Dist Image, dist_transform time = %llu\n",(t1-t0));
    imshow("Sure FG",sure_fg);
    printf("FG Image\n");

    // Find unknown region
    Mat unknown;
    sure_bg.convertTo(sure_bg, CV_8U);
    sure_fg.convertTo(sure_fg, CV_8U);
    subtract(sure_bg,sure_fg,unknown);
    // imshow("unknown",unknown);
    printf("Unknown Image\n");
    
    // Marker Labling
    Mat markers;
    Mat disp_markers;
    connectedComponents(sure_fg,markers);
    markers = markers+1;
    markers.convertTo(disp_markers, CV_8UC1, 1, 0);
    minMaxLoc(disp_markers, &min, &max); 
    applyColorMap(disp_markers/max, disp_markers, cv::COLORMAP_JET);
    imshow("Markers", disp_markers);
    printf("Markers Image\n");

    // Now, mark the region of unknown with zero
    Mat mask;
    inRange(unknown, 255, 255, mask);
    markers.setTo(0, mask);

    // Watershed
    watershed(src, markers);
    Mat mask2;
    inRange(markers, -1, -1, mask2);
    src.setTo(Scalar(255, 0, 0), mask2);
    markers.convertTo(disp_markers, CV_8UC1, 1, 0); 
    applyColorMap(disp_markers*8, disp_markers, cv::COLORMAP_JET);
    imshow("Watershed Markers", disp_markers);
    printf("Watershed Markers\n");

    // Visualize the final image
    imshow("Final Result", src);
    printf("Final Image\n");
    waitKey(100000);
    return 0;
}