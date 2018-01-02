#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class RMD
{
private:
	Mat cur_frame;					//Holds current frame
	Mat gray_frame;					//Holds current frame(gray)

public:
	Mat WarpImage(Mat& img);
};