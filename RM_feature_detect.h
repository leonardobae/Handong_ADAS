#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>

#define num_sub_roi_x 4
#define num_sub_roi_y 2

//GFTTDetector parameter
#define maxCorners 100
#define qualityLevel 0.005
#define minDistance 4
#define blockSize 2
#define querySize 4

//Orb Size
#define edgeThreshold 2
#define patchSize 2

using namespace std;
using namespace cv;

class RM
{
private:
	vector<Mat> candidateBoxes;					//Contains candidate boundedRect Mats that contains roadmark.
	vector<Mat> selectedBoxes;			//
	int candidateBox_cnt = 0;
	bool window_open_flag = false;
	int no_candidate_cnt = 0;
	int selectedBox_cnt = 0;

	//DB
	vector<Mat> DBdescriptorVector;
	vector<Mat> DBImgVector;
	vector<vector<KeyPoint>> DBKeypointVector;

	//QUERY 
	vector<Mat> candidateDescriptorVector;
	vector<vector<KeyPoint>> candidateKeypointVector;
	vector<KeyPoint> keypoints;
	vector<Vec4i> hierarchy;

public:
	RM();
	void subRoiBinary(Mat& img);
	void widthBinary(Mat& img);
	void imgThresh(Mat& img);
	Mat WarpImage(Mat& img);    //*****main으로 matrix를 반환하도록 변경
	void HarrisCorner(Mat& img);
	int Cont(Mat& img);			//*****main에서 호출 하고 candidateBoxes.size()를 반환하도록 변경. window_open_flag1,2 관리 
	void fast_detection(Mat& img);
	void ChooseCanditates(Mat& roi_binary);
	void MyFeatureDetector();
	void SetDB();				//******setQueryData 함수 이름만 바꿈!
	int DescriptorMatching();
	void ShowImage(String windowName, Mat frame);
	int VectorClear();

	Mat c_frame; //current frame
	Mat t_frame;
	Mat w_frame;
	Mat roi;
	Mat HC;
	Mat FC;
	Mat warpImg;
	vector<vector<Point>> contours; // 컨투어 
	vector<Point> approxCurve;
	bool window_open_flag2 = false;
	

};