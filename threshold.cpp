#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#define SQR(x) ((x)*(x))

using namespace std;
using namespace cv;

double slope(Vec4i line);

int t = 0;
int gmax_val = 0;
bool flag = false;
Mat grayimg2;
int bkj_flag = 0;
bool do_it = false;
int c_point = 0;



vector<Mat> candidateBoxes;
int no_candidate_cnt = 0;
bool window_open_flag = false;

void thinningIteration(cv::Mat& img, int iter)
{
	CV_Assert(img.channels() == 1);
	CV_Assert(img.depth() != sizeof(uchar));
	CV_Assert(img.rows > 3 && img.cols > 3);

	cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

	int nRows = img.rows;
	int nCols = img.cols;

	if (img.isContinuous()) {
		nCols *= nRows;
		nRows = 1;
	}

	int x, y;
	uchar *pAbove;
	uchar *pCurr;
	uchar *pBelow;
	uchar *nw, *no, *ne;    // north (pAbove)
	uchar *we, *me, *ea;
	uchar *sw, *so, *se;    // south (pBelow)

	uchar *pDst;

	// initialize row pointers
	pAbove = NULL;
	pCurr = img.ptr<uchar>(0);
	pBelow = img.ptr<uchar>(1);

	for (y = 1; y < img.rows - 1; ++y) {
		// shift the rows up by one
		pAbove = pCurr;
		pCurr = pBelow;
		pBelow = img.ptr<uchar>(y + 1);

		pDst = marker.ptr<uchar>(y);

		// initialize col pointers
		no = &(pAbove[0]);
		ne = &(pAbove[1]);
		me = &(pCurr[0]);
		ea = &(pCurr[1]);
		so = &(pBelow[0]);
		se = &(pBelow[1]);

		for (x = 1; x < img.cols - 1; ++x) {
			// shift col pointers left by one (scan left to right)
			nw = no;
			no = ne;
			ne = &(pAbove[x + 1]);
			we = me;
			me = ea;
			ea = &(pCurr[x + 1]);
			sw = so;
			so = se;
			se = &(pBelow[x + 1]);

			int A = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
				(*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
				(*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
				(*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
			int B = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
			int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
			int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
				pDst[x] = 1;  //set white
		}
	}

	img &= ~marker;
}

/**
* Function for thinning the given binary image
*
* Parameters:
*       src  The source image, binary with range = [0,255]
*       dst  The destination image
*/
void thinning(const cv::Mat& src, cv::Mat& dst)
{
	dst = src.clone();
	dst /= 255;         // convert to binary image

	cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
	cv::Mat diff;

	do {
		thinningIteration(dst, 0); //step1 
		thinningIteration(dst, 1); //step2
		cv::absdiff(dst, prev, diff);
		dst.copyTo(prev);
	} while (cv::countNonZero(diff) > 0);

	dst *= 255;
}
/*Function to find the center of mass*/
int findCenter(cv::Point2f &center, cv::Mat &rmImg)
{
	int nCount = 0;
	int nSumX = 0;
	int nSumY = 0;
	Mat cImg = rmImg.clone();

	for (int i = 0; i < rmImg.rows; i++)
	{
		for (int j = 0; j < rmImg.cols; j++)
		{
			if (rmImg.at<uchar>(i, j) == 255)
			{
				nCount++;
				nSumX += j;
				nSumY += i;

			}
		}
	}
	/////////////////////////////////////
	if (nCount > 0)
	{
		center.x = (float)nSumX / (float)nCount;
		center.y = (float)nSumY / (float)nCount;
	}
	else
		center.x = center.y = 0.0f;

	std::cout << "center (x, y) = (" << center.x << ", " << center.y << ")" << std::endl;
	cv::circle(cImg, center, 1, cv::Scalar(255), 2, 8, 0);

	imshow("circle", cImg);
	return 0;
}

int findLine(cv::Mat& rmImg, Vec4i& verticalL)
{
	int nLine = 0;
	Mat cLineImg(rmImg.size(), CV_8UC1);

	cLineImg = rmImg.clone();


	//HoughLineP 를 이용한 line 찾기 
	vector<Vec4i> lines;

	HoughLinesP(cLineImg, lines, 1, CV_PI / 180, 10, 20, 5); //patameter (input img, line, rho, theta, threshhold, minLineLength, maxLineGap ) 라인을 많이 찾으면 좋을 거같아서 임의로 정한값.
	cout << "Lines.size() = " << lines.size() << endl;


	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		double s = slope(l);   //call the fuction for computing the slope of line   
		//Draw all line
		//line(cLineImg, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 255, 255), 2, CV_AA); 
		//cout << i << "th line of slope = " << s << endl; //debugging
		if (abs(s) > 85 && abs(s) < 95)      //수직을 찾기 위해 임의로 정한값 
		{
			verticalL = l;
			line(cLineImg, Point(verticalL[0], verticalL[1]), Point(verticalL[2], verticalL[3]), Scalar(255, 255, 255), 2, CV_AA);  //line 함수 시작점과 끝점 포인트 2개 
		}
	}
	//cout << "verticalL[0] = " <<verticalL[0] << endl;
	imshow("line", cLineImg);
	return 0;
}

double slope(Vec4i line) /*Funtion to compute theta */
{
	double Angle = atan2(line[3] - line[1], line[2] - line[0]) * 180.0 / CV_PI; //atan 역탄젠트
	//cout << "Angle = " << Angle << endl;

	return Angle;
}

void recognizeRM(Point2f &center, Vec4i verticalL)
{
	int middleX = (verticalL[0] + verticalL[2]) / 2; //고민고민
	float ndiff = 1.5f;
	cout << "middleX = " << middleX << endl;

	//recogize RoadMark
	if (middleX == 0)
	{
		cout << "/////////////////////////stopLine soon///////////////////////" << endl;
	}
	else if (abs(center.x - (float)middleX) < ndiff)
	{
		cout << "/////////////////////////go straight//////////////////////////" << endl;

	}
	else if (center.x > middleX)
	{
		cout << "/////////////////////////right//////////////////////////" << endl;
	}
	else if (center.x < middleX)
	{
		cout << "/////////////////////////left//////////////////////////" << endl;
	}

	return;
}

void bkj_3(int i, int j, int diff)
{
	for (; j < grayimg2.cols - 5; j++)
	{
		diff = grayimg2.at<uchar>(i, j) - grayimg2.at<uchar>(i, j + 1);
		diff = abs(diff);

		if (diff < 15)
		{
			diff = grayimg2.at<uchar>(i, j + 1) - grayimg2.at<uchar>(i, j + 2);
			diff = abs(diff);

			if (diff < 15)
			{
				diff = grayimg2.at<uchar>(i, j + 2) - grayimg2.at<uchar>(i, j + 3);
				diff = abs(diff);

				if (diff < 15)
				{
					do_it = true;
					break;
				}
			}
		}
	}
}

void bkj_2(Mat _grayimg)
{
	Mat src = _grayimg;
	int width = src.cols;
	int height = src.rows;
	int i, j = 0;
	bool flaggg = false;

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
				bkj_3(i, j, diff);
				flaggg = true;
				break;
			}
		}
		if (flaggg)
			break;
	}
}


int drawPeaks(Mat &histImage, vector<int>& peaks, int hist_size = 256, Scalar color = Scalar(0, 0, 255))
{
	int bin_w = cvRound((double)histImage.cols / hist_size);
	for (size_t i = 0; i < peaks.size(); i++)
		line(histImage, Point(bin_w * peaks[i], histImage.rows), Point(bin_w * peaks[i], 0), color);

	line(histImage, Point(bin_w * 170, histImage.rows), Point(bin_w * 170, 0), Scalar(255, 0, 0));

	imshow("Peaks", histImage);
	return EXIT_SUCCESS;
}

Mat drawHistogram(Mat &hist, int hist_h = 400, int hist_w = 1024, int hist_size = 256, Scalar color = Scalar(255, 255, 255), int type = 2)
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

PeakInfo peakInfo(int pos, int left_size, int right_size, float value)
{
	PeakInfo output;
	output.pos = pos;
	output.left_size = left_size;
	output.right_size = right_size;
	output.value = value;
	return output;
}

vector<PeakInfo> findPeaks(InputArray _src, int window_size)
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
		if (src2.at<float>(i) < src2.at<float>(i+1))
		{
			gmax_val = i+1;                                  //극대값 gmax_val
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

int bkj_t(InputArray _src, int _g_max, int _peaks1, vector<PeakInfo> _peak_arr)                     //adaptive하게 threshold를 찾는다.
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

vector<int> getLocalMaximum(InputArray _src, int smooth_size = 9, int neighbor_size = 3, float peak_per = 0.5) //if you play with the peak_per attribute value, you can increase/decrease the number of peaks found
{
	flag = false;         //flag false로 initialize
	Mat src = _src.getMat().clone();

	vector<int> output;
	GaussianBlur(src, src, Size(smooth_size, smooth_size), 0);
	vector<PeakInfo> alt_peaks;
	vector<PeakInfo> peaks = findPeaks(src, neighbor_size);

	
	if (peaks.size() == 0)
	{
		bkj_2(grayimg2);   //요기
		alt_peaks = findPeaks(src, neighbor_size);

		if (alt_peaks.size())
		{
			int index = alt_peaks.size() - 1;
			flag = true;
			t = bkj_t(src, gmax_val, alt_peaks[index].pos, alt_peaks);
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

	//t = bkj_t(src, gmax_val, peaks1, peaks);

	//cout << "adpative_t : " << t << endl;

	if (peaks.size())
	{
		flag = true;
		t = bkj_t(src, gmax_val, peaks[0].pos, peaks);

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



void Cont(Mat& img)
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
	if (window_open_flag == true && no_candidate_cnt > 10){ // //window가 열려있을 때 10frame 이상 candidate 못찾으면 창 닫음
		window_open_flag = false;
		candidateBoxes.clear();
		destroyWindow("dstImage");
	}

}

Mat WarpImage(Mat& ROI_src);
Mat Histog(Mat frame);


Mat c_frame;
double fr_cnt = 0;
Mat grayimg;

int main()
{
	Mat frame;
	Mat warpImg;
	bool stop = false;
	char key;
	Mat thresh_bkj;

	Mat histogram;
	float channel_range[] = { 0, 256 };
	const float* channel_ranges = { channel_range };
	int number_bins = 256;
	bool uniform = true;
	bool accumulate = false;

	VideoCapture vid("good5.mpg");

	if (!vid.isOpened()) {
		cerr << "Video File Open Error" << endl;
		exit(1);
	}

	while (1)
	{
		if (!stop)
		{
			if (!vid.read(frame))
				break;

			fr_cnt = vid.get(CV_CAP_PROP_POS_FRAMES);

			frame.copyTo(c_frame);

			imshow("Original", frame);
			warpImg = WarpImage(c_frame(Rect(480, 450, 380, 200))); //******warp한 이미지 반환후 cont 함수 인자로 넣어서 메인에서 call

			cvtColor(warpImg, grayimg, CV_BGR2GRAY);

			//threshold(grayimg, thresh_otsu, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

			imshow("grayimg", grayimg);

			grayimg.copyTo(grayimg2);               //bkj_2에서 쓸 회색이미지 복사

			calcHist(&grayimg, 1, 0, Mat(), histogram, 1, &number_bins, &channel_ranges, uniform, accumulate);

			vector<int> peaks = getLocalMaximum(histogram);

			if (flag)
			{
				threshold(grayimg, thresh_bkj, t, 255, CV_THRESH_BINARY);
				imshow("bkj_t", thresh_bkj);
				Cont(thresh_bkj);
			}

			//imshow("g_Histogram", Histog(grayimg));
			//imshow("norm_histog", Histog(norm));

			//if (RMclass.Cont(warpImg) > 0){  //*****return값: candidateBoxes.size() > 0 
			//   RMclass.MyFeatureDetector();
			//   RMclass.DescriptorMatching();
			//RMclass.VectorClear();
			//}
			//RMclass.VectorClear();
		}

		key = waitKey(CV_CAP_PROP_FPS);

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

Mat WarpImage(Mat& ROI_src)
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
		circle(c_frame(Rect(480, 450, 380, 200)), corners[i], 3, Scalar(0, 0, 255), 3);

	imshow("X, Y Points", c_frame(Rect(480, 450, 380, 200)));

	return warpImg;
}


Mat Histog(Mat frame)
{
	Mat greyImg = frame;
	Mat histogram;
	const int* channel_numbers = { 0 };
	float channel_range[] = { 0, 256 };
	const float* channel_ranges = { channel_range };
	int number_bins = 256;
	bool uniform = true;
	bool accumulate = false;

	calcHist(&greyImg, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges, uniform, accumulate);

	// Plot the histogram
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / number_bins);

	Mat histImage(hist_h, hist_w, CV_8UC1);
	normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, CV_32F);

	for (int i = 1; i < number_bins; i++)
	{

		rectangle(histImage, Point(bin_w*(i), histImage.rows), Point((i + 1)*bin_w, histImage.rows - cvRound(histogram.at<float>(i))), Scalar(0), -1);
		/*line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
		Point(bin_w*(i), hist_h - cvRound(histogram.at<float>(i))),
		Scalar(255, 0, 0), 2, 8, 0);*/
	}

	return histImage;
}