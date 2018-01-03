#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

struct Length
{
	int pos1;
	int pos2;
	int size()
	{
		return pos2 - pos1 + 1;
	}
	Length()
	{
		pos1 = 0;
		pos2 = 0;
	}
};

struct PeakInfo
{
	int pos;         //pos 안에 최고봉우리, 즉 index값(x축값)을 가지고 있음.
	int left_size;      //최고봉우리 왼쪽의 index값
	int right_size;      //최고봉우리 오른쪽의 index값
	float value;      //최고봉우리의 pixel value
};

class RMD
{
private:
	

public:
	/* Variables */

	Mat cur_frame;										//Holds current frame
	Mat warp_img;										//Holds the warped image
	Mat gray_frame;										//Holds the warped image(gray)
	Mat binary_frame;									//Holds warped image(binary)
	
	bool flag = false;									//Flag for loop termination
	bool do_it = false;									//Flag for threshold initiation
	int gmax_val;										//Holds the global maximum value of pixels
	int thresh_val;										//Holds the adaptive threshold value
	int c_point;

	bool window_open_flag = false;						//Flag for opening window of candidates
	vector<Mat> candidateBoxes;							//Vector that holds road mark candidates
	int no_candidate_cnt;								//Counting number of candidates



	/* Functions */

	Mat WarpImage(Mat& img);							//Warps image
	void bkj_2(Mat _grayimg);
	int drawPeaks(Mat &histImage, vector<int>& peaks, int hist_size, Scalar color);
	Mat drawHistogram(Mat &hist, int hist_h, int hist_w, int hist_size, Scalar color, int type);
	PeakInfo peakInfo(int pos, int left_size, int right_size, float value);
	vector<PeakInfo> findPeaks(InputArray _src, int window_size);
	int dynamicThreshold(InputArray _src, int _g_max, int _peaks1, vector<PeakInfo> _peak_arr);
	vector<int> getLocalMaximum(InputArray _src, int , int , float);
	void Cont(Mat& img);
};