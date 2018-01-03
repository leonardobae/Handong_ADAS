#pragma once
#include "RMD.h"
#include <vector>


void RMD::bkj_2(Mat _grayimg)
{
	Mat src = _grayimg;
	int i, j = 0;
	bool flag = false;

	int diff = 0;

	for (i = 0; i < src.rows; i++)
	{
		for (j = 0; j < src.cols - 4; j++)
		{
			diff = src.at<uchar>(i, j) - src.at<uchar>(i, j + 1);      //diff = f(x, y) - f(x + 1, y) ,convolution(?)
			diff = abs(diff);                                          //절대값 취한다.

			if (diff > 18)
			{
				c_point = src.at<uchar>(i, j + 2);
				//cout << "c_point: " << c_point << endl;
				diff = gray_frame.at<uchar>(i, j) - gray_frame.at<uchar>(i, j + 1);
				diff = abs(diff);

				if (diff < 15)
				{
					diff = gray_frame.at<uchar>(i, j + 1) - gray_frame.at<uchar>(i, j + 2);
					diff = abs(diff);

					if (diff < 15)
					{
						diff = gray_frame.at<uchar>(i, j + 2) - gray_frame.at<uchar>(i, j + 3);
						diff = abs(diff);

						if (diff < 15)
						{
							do_it = true;
							break;
						}
					}
				}
				flag = true;
				break;
			}
		}
		if (flag)
			break;
	}
}


int RMD::drawPeaks(Mat &histImage, vector<int>& peaks, int hist_size = 256, Scalar color = Scalar(0, 0, 255))
{
	int bin_w = cvRound((double)histImage.cols / hist_size);
	for (size_t i = 0; i < peaks.size(); i++)
		line(histImage, Point(bin_w * peaks[i], histImage.rows), Point(bin_w * peaks[i], 0), color);

	line(histImage, Point(bin_w * 170, histImage.rows), Point(bin_w * 170, 0), Scalar(255, 0, 0));

	imshow("Peaks", histImage);
	return EXIT_SUCCESS;
}

Mat RMD::drawHistogram(Mat &hist, int hist_h = 400, int hist_w = 1024, int hist_size = 256, Scalar color = Scalar(255, 255, 255), int type = 2)
{
	int bin_w = cvRound((double)hist_w / hist_size);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	switch (type) {
	case 1:
		for (int i = 0; i < histImage.cols; i++)
		{
			const unsigned x = i;
			const unsigned y = hist_h;

			line(histImage, Point(bin_w * x, y),
				Point(bin_w * x, y - cvRound(hist.at<float>(i))),
				color);
		}

		break;
	case 2:
		for (int i = 1; i < hist_size; ++i)
		{
			Point pt1 = Point(bin_w * (i - 1), hist_h);
			Point pt2 = Point(bin_w * i, hist_h);
			Point pt3 = Point(bin_w * i, hist_h - cvRound(hist.at<float>(i)));
			Point pt4 = Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1)));
			Point pts[] = { pt1, pt2, pt3, pt4, pt1 };

			fillConvexPoly(histImage, pts, 5, color);
		}
		break;
	default:
		for (int i = 1; i < hist_size; ++i)
		{
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
				color, 1, 8, 0);
		}

		break;
	}
	return histImage;
}


PeakInfo RMD::peakInfo(int pos, int left_size, int right_size, float value)
{
	PeakInfo output;
	output.pos = pos;
	output.left_size = left_size;
	output.right_size = right_size;
	output.value = value;
	return output;
}

vector<PeakInfo> RMD::findPeaks(InputArray _src, int window_size)
{
	gmax_val = 1;                  //gmax_val 0으로 초기화
	Mat src = _src.getMat();

	Mat slope_mat = src.clone();

	// Transform initial matrix into 1channel, and 1 row matrix
	Mat src2 = src.reshape(1, 1);

	int size = window_size / 2;

	Length up_hill, down_hill;
	vector<PeakInfo> output;

	int pre_state = 0;
	int i = size;

	for (size_t i = 0; i < src2.cols - 2; i++)
	{
		if (src2.at<float>(i) < src2.at<float>(i + 1))
		{
			gmax_val = i + 1;                               
		}
	}
	
	while (i < src2.cols - size)
	{
		float cur_state = src2.at<float>(i + size) - src2.at<float>(i - size);

		if (cur_state > 0)
			cur_state = 2;
		else if (cur_state < 0)
			cur_state = 1;
		else cur_state = 0;

		// In case you want to check how the slope looks like
		slope_mat.at<float>(i) = cur_state;

		if (pre_state == 0 && cur_state == 2)
			up_hill.pos1 = i;
		else if (pre_state == 2 && cur_state == 1)
		{
			up_hill.pos2 = i - 1;
			down_hill.pos1 = i;
		}

		if ((pre_state == 1 && cur_state == 2) || (pre_state == 1 && cur_state == 0))
		{
			down_hill.pos2 = i - 1;
			int max_pos = up_hill.pos2;   // 봉우리 detect하고 max_pos는 지금 detect한 시점에서의 봉우리의 i값(x축 값)을 가지고 있음.
			if (src2.at<float>(up_hill.pos2) < src2.at<float>(down_hill.pos1))
				max_pos = down_hill.pos1;    // pre_state == 2 && cur_state == 2 에 대한 조건 대신 이걸 씀. 결국 최고봉우리값 detect하기 위해서.

			if (src2.at<float>(gmax_val) < src2.at<float>(max_pos))
			{
				gmax_val = max_pos;                                  //극대값 gmax_val
			}

			if (max_pos > 170 && (src2.at<float>(max_pos) > 145 && src2.at<float>(max_pos)< 1300))         // 1차 필터 (index > 170 and value >145) 이어야 vector에 push.
			{
				PeakInfo peak_info = peakInfo(max_pos, up_hill.size(), down_hill.size(), src2.at<float>(max_pos));

				output.push_back(peak_info);
			}

			if (do_it == true)
			{
				if (max_pos > gmax_val && max_pos < c_point + 20 && src2.at<float>(max_pos) < 1300 && max_pos <= 170)
				{
					PeakInfo peak_info = peakInfo(max_pos, up_hill.size(), down_hill.size(), src2.at<float>(max_pos));

					output.push_back(peak_info);
				}
			}
		}
		i++;
		pre_state = (int)cur_state;
	}
	do_it = false;
	return output;
}

int RMD::dynamicThreshold(InputArray _src, int _g_max, int _peaks1, vector<PeakInfo> _peak_arr)                     //adaptive하게 threshold를 찾는다.
{
	int lowest = _g_max;
	int i;
	Mat src = _src.getMat();
	Mat src2 = src.reshape(1, 1);

	for (i = _g_max; i < _peaks1; i++)
	{
		if (src2.at<float>(i) < src2.at<float>(lowest))
		{
			lowest = i;
		}
	}

	return lowest;
}

vector<int> RMD::getLocalMaximum(InputArray _src, int smooth_size = 9, int neighbor_size = 3, float peak_per = 0.5) //if you play with the peak_per attribute value, you can increase/decrease the number of peaks found
{
	flag = false;         //flag false로 initialize
	Mat src = _src.getMat().clone();

	vector<int> output;
	GaussianBlur(src, src, Size(smooth_size, smooth_size), 0);
	vector<PeakInfo> alt_peaks;
	vector<PeakInfo> peaks = findPeaks(src, neighbor_size);


	if (peaks.size() == 0)
	{
		bkj_2(gray_frame);   //요기
		alt_peaks = findPeaks(src, neighbor_size);

		if (alt_peaks.size())
		{
			int index = alt_peaks.size() - 1;
			flag = true;
			thresh_val = dynamicThreshold(src, gmax_val, alt_peaks[index].pos, alt_peaks);
		}
	}

	//cout << "peaks.size() = " << peaks.size() << endl;
	//cout << "alt_peaks.size() = " << alt_peaks.size() << endl;
	//cout << "---------------------------" << endl;


	/*for (size_t i = 0; i < peaks.size(); i++)
	{
	cout << "index of peaks[" << i << "] = " << peaks[i].pos << endl;
	cout << "value of peaks[" << i << "] = " << peaks[i].value << endl;
	}

	for (size_t i = 0; i < alt_peaks.size(); i++)
	{
	cout << "index of peaks[" << i << "] = " << alt_peaks[i].pos << endl;
	cout << "value of peaks[" << i << "] = " << alt_peaks[i].value << endl;
	}*/

	//cout << "---------------------------" << endl;

	//t = dynamicThreshold(src, gmax_val, peaks1, peaks);

	//cout << "adpative_t : " << t << endl;

	if (peaks.size())
	{
		flag = true;
		thresh_val = dynamicThreshold(src, gmax_val, peaks[0].pos, peaks);

		for (size_t i = 0; i < peaks.size(); i++)
		{
			output.push_back(peaks[i].pos);
		}
	}

	if (peaks.size() == 0)
	{
		for (size_t i = 0; i < alt_peaks.size(); i++)
		{
			output.push_back(alt_peaks[i].pos);
		}
	}

	Mat histImg = drawHistogram(src);
	drawPeaks(histImg, output);
	return output;

}



void RMD::Cont(Mat& img)
{
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	vector<Point> approxCurve;
	RNG rng(12345);
	Mat output;
	Point pt;

	pt.x = 5; //
	pt.y = 5; //

	Mat temp = Mat::zeros(img.rows + 40, img.cols + 40, CV_8UC1);
	img.copyTo(temp(cv::Rect(20, 20, img.cols, img.rows)));

	findContours(temp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	//cout << "contour size : " << contours.size() << endl;

	vector<Rect> boundRect(contours.size());

	for (size_t i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), approxCurve, arcLength(Mat(contours[i]), true)*0.01, true);
		boundRect[i] = boundingRect(Mat(approxCurve));

		if (boundRect[i].height > boundRect[i].width && boundRect[i].height < 4 * boundRect[i].width && boundRect[i].height > 110 && boundRect[i].width < 170)
		{ // RoadMark Candidate condition - 임의로 설정한 값임.

		  //Mat temp = Mat::zeros(boundRect[i].height + 20, boundRect[i].width + 20, CV_8UC1);
		  //Mat temp = Mat::zeros(t_frame.cols + 300, t_frame.rows + 300, CV_8UC1);
			Mat c_box;
			Mat thin;

			drawContours(temp, vector<vector<Point>>(1, approxCurve), 0, Scalar(255), CV_FILLED);
			//drawContours(temp, vector<vector<Point>>(1, approxCurve), 0, Scalar(255), 2, 8, hierarchy, 0, Point());

			boundRect[i].height += 10;
			boundRect[i].width += 10;
			boundRect[i] -= pt;
			//rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(255), 2);

			c_box = temp(boundRect[i]);
			resize(c_box, c_box, Size(c_box.cols / 2, c_box.rows / 2));
			candidateBoxes.push_back(c_box);

			imshow("Candidates", c_box);

			/*thin = c_box.clone();

			thinning(thin, thin);

			Point2f center;
			findCenter(center, thin);

			Vec4i verticalL;
			findLine(thin, verticalL);

			recognizeRM(center, verticalL);

			imshow("Thinning", thin);*/

		}
		rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 255, 255), 2);
	}

	//cout << "candidateBoxes.size() = " << candidateBoxes.size() << endl;
	//cout << "no_candidate_cnt = " << no_candidate_cnt << endl;

	if (candidateBoxes.size() > 0)
	{
		no_candidate_cnt = 0; //initialize
	}
	else
	{
		no_candidate_cnt++;
	}
	if (window_open_flag == true && no_candidate_cnt > 10) { // //window가 열려있을 때 10frame 이상 candidate 못찾으면 창 닫음
		window_open_flag = false;
		candidateBoxes.clear();
		destroyWindow("dstImage");
	}

}


Mat RMD::WarpImage(Mat& ROI_src)
{
	vector<Point2f> corners(4);
	corners[0] = Point2f(150, 40);
	corners[1] = Point2f(278, 40);
	corners[2] = Point2f(53, 170);
	corners[3] = Point2f(365, 170);
	Size warpSize(250, 385);
	Mat warpImg(warpSize, ROI_src.type());

	//Warping 후의 좌표
	vector<Point2f> warpCorners(4);
	warpCorners[0] = Point2f(0.0f, 0.0f);
	warpCorners[1] = Point2f(warpImg.cols, 0.0f);
	warpCorners[2] = Point2f(0.0f, warpImg.rows);
	warpCorners[3] = Point2f(warpImg.cols, warpImg.rows);

	//Transformation Matrix 구하기
	Mat trans = getPerspectiveTransform(corners, warpCorners);

	//Warping
	warpPerspective(ROI_src, warpImg, trans, warpSize);

	//Cont(warpImg);

	for (int i = 0; i<corners.size(); i++)
		circle(cur_frame(Rect(480, 450, 380, 200)), corners[i], 3, Scalar(0, 0, 255), 3);

	imshow("X, Y Points", cur_frame(Rect(480, 450, 380, 200)));

	return warpImg;
}

