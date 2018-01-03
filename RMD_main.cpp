#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "RMD.h"


using namespace cv;
using namespace std;

double fr_cnt = 0;

int main()
{
	RMD RMD;
	string vid_name;
	bool stop = false;
	char key;	

	/*---------Variables for histogram---------*/
	Mat histogram;
	float channel_range[] = { 0, 256 };
	const float* channel_ranges = { channel_range };
	int number_bins = 256;
	bool uniform = true;
	bool accumulate = false;
	/*----------------------------------------*/


	cout << "Enter video name: ";
	getline(cin, vid_name);
	VideoCapture vid(vid_name);

	/*-----Variables for video information-----*/
	float _fps = vid.get(CV_CAP_PROP_FPS);
	int _cols = vid.get(CV_CAP_PROP_FRAME_WIDTH);
	int _rows = vid.get(CV_CAP_PROP_FRAME_HEIGHT);
	double _nframe = vid.get(CV_CAP_PROP_FRAME_COUNT);
	Size _ImageSize = Size(_cols, _rows);
	/*-----------------------------------------*/

		
	if (!vid.isOpened()) {
		cerr << "Video File Open Error" << endl;
		exit(1);
	}
	
	/*-------------Output video information-------------------*/
	cout << endl;
	cout << "Input Video File Name : " << "test2" << endl;
	cout << "Frame Per Seconds : " << _fps << endl;
	cout << "Frame Size : " << _cols << " x " << _rows << endl;
	cout << "Frame Count : " << _nframe << endl;
	/*--------------------------------------------------------*/


	while (1)
	{
		if (!stop)
		{
			if (!vid.read(RMD.cur_frame))
				break;
			fr_cnt = vid.get(CV_CAP_PROP_POS_FRAMES);
			
			imshow("Original", RMD.cur_frame);												//display current frame

			/* --<Warping Rect Information>-- */
			//1. good4.mpg = 

			/*--------------------------------*/

			RMD.warp_img = RMD.WarpImage(RMD.cur_frame(Rect(480, 450, 380, 200)));
			imshow("Warp", RMD.warp_img);

			cvtColor(RMD.warp_img, RMD.gray_frame, CV_BGR2GRAY);

			calcHist(&RMD.gray_frame, 1, 0, Mat(), histogram, 1, &number_bins, &channel_ranges, uniform, accumulate);

			vector<int> peaks = RMD.getLocalMaximum(histogram, 9, 3, 0.5);

			if (RMD.flag)
			{
				threshold(RMD.gray_frame, RMD.binary_frame, RMD.thresh_val, 255, CV_THRESH_BINARY);
				imshow("Threshold", RMD.binary_frame);
				RMD.Cont(RMD.binary_frame);
			}

		}

		key = waitKey(1000/_fps);

		if (key == 27)
			break;
		else if (key == 32)
		{
			if (stop == true)
				stop = false;
			else if (stop == false)
				stop = true;
		}
		else if (key == ']')
		{
			fr_cnt += 100;
			vid.set(CV_CAP_PROP_POS_FRAMES, fr_cnt - 1);
		}
		else if (key == '[')
		{
			fr_cnt -= 100;
			vid.set(CV_CAP_PROP_POS_FRAMES, fr_cnt - 1);
		}
		else if (key == '.')
		{
			fr_cnt += 10;
			vid.set(CV_CAP_PROP_POS_FRAMES, fr_cnt - 1);
		}
		else if (key == ',')
		{
			fr_cnt -= 10;
			vid.set(CV_CAP_PROP_POS_FRAMES, fr_cnt - 1);
		}
	}

	waitKey(0);
	return 0;
}