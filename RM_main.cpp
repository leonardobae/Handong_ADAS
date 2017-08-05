#include "RM_feature_detect.h"

double fr_cnt = 0;

int main()
{
	RM RMclass;
	VideoCapture vid("test3.mpg");

	if (!vid.isOpened()) {
		cerr << "Video File Open Error" << endl;
		exit(1);
	}

	/*동영상 정보 출력*/
	float _fps = vid.get(CV_CAP_PROP_FPS);
	int _cols = vid.get(CV_CAP_PROP_FRAME_WIDTH);
	int _rows = vid.get(CV_CAP_PROP_FRAME_HEIGHT);
	int _nframe = vid.get(CV_CAP_PROP_FRAME_COUNT);
	Size _ImageSize = Size(_cols, _rows);


	cout << endl;
	cout << "Input Video File Name : " << "test2" << endl;
	cout << "Frame Per Seconds : " << _fps << endl;
	cout << "Frame Size : " << _cols << " x " << _rows << endl;
	cout << "Frame Count : " << _nframe << endl;

	/*각 프레임에 ROI 씌우기*/

	Mat frame;
	bool stop = false;
	char key;

	while (1)
	{
		if (!stop)
		{
			if (!vid.read(frame))
				break;
			fr_cnt = vid.get(CV_CAP_PROP_POS_FRAMES);

			frame.copyTo(RMclass.c_frame);

			imshow("Original", frame);

			RMclass.WarpImage(RMclass.c_frame(Rect(480, 450, 380, 200)));

			RMclass.MyFeatureDetector();
			RMclass.DescriptorMatching();

		}

		key = waitKey(30);

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

}