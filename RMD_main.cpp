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

	cout << "Enter video name: ";
	getline(cin, vid_name);

	VideoCapture vid(vid_name);

	if (!vid.isOpened()) {
		cerr << "Video File Open Error" << endl;
		exit(1);
	}

	/*동영상 정보 출력*/
	float _fps = vid.get(CV_CAP_PROP_FPS);
	int _cols = vid.get(CV_CAP_PROP_FRAME_WIDTH);
	int _rows = vid.get(CV_CAP_PROP_FRAME_HEIGHT);
	double _nframe = vid.get(CV_CAP_PROP_FRAME_COUNT);
	Size _ImageSize = Size(_cols, _rows);

	cout << endl;
	cout << "Input Video File Name : " << "test2" << endl;
	cout << "Frame Per Seconds : " << _fps << endl;
	cout << "Frame Size : " << _cols << " x " << _rows << endl;
	cout << "Frame Count : " << _nframe << endl;

	/*각 프레임에 ROI 씌우기*/
	Mat frame;
	Mat warpImg;
	bool stop = false;
	char key;

	while (1)
	{
		if (!stop)
		{
			if (!vid.read(frame))
				break;
			fr_cnt = vid.get(CV_CAP_PROP_POS_FRAMES);

			imshow("Original", frame);

		}

		key = waitKey(1000/_fps);

		if (key == 27)
			break;
		else if (key == 32) {
			if (stop == true)

				stop = false;
			else if (stop == false)
				stop = true;
		}
		else if (key == ']') {
			fr_cnt += 100;
			vid.set(CV_CAP_PROP_POS_FRAMES, fr_cnt - 1);
		}
		else if (key == '[') {
			fr_cnt -= 100;
			vid.set(CV_CAP_PROP_POS_FRAMES, fr_cnt - 1);
		}
		else if (key == '.') {
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