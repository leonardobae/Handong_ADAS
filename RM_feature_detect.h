#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>

#define num_sub_roi_x 4
#define num_sub_roi_y 2

//GFTTDetector parameter
#define maxCorners 30
#define qualityLevel 0.01
#define minDistance 5
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
	vector<Mat> candidateBoxes;
	vector<Mat> selectedBoxes;
	int candidateBox_cnt = 0;
	bool window_open_flag = false;
	int no_candidate_cnt = 0;
	int selectedBox_cnt = 0;

	//DB
	vector<Mat> descriptorVector;
	vector<Mat> DBImgVector;
	vector<vector<KeyPoint>> DBKeypointVector;

	//QUERY 
	vector<Mat> candidateDescriptorVector;
	vector<vector<KeyPoint>> candidadeteKeypointVector;
	vector<KeyPoint> keypoints;
	vector<Vec4i> hierarchy;

public:
	RM();
	void subRoiBinary(Mat& img);
	void widthBinary(Mat& img);
	void imgThresh(Mat& img);
	void WarpImage(Mat& img);
	void HarrisCorner(Mat& img);
	void Cont(Mat& img);
	void fast_detection(Mat& img);
	void ChooseCanditates(Mat& roi_binary);
	void MyFeatureDetector();
	void SetQueryData();
	int DescriptorMatching();
	void ShowImage(String windowName, Mat frame);

	Mat c_frame; //current frame
	Mat t_frame;
	Mat w_frame;
	Mat roi;
	Mat HC;
	Mat FC;
	Mat warpImg;
	vector<vector<Point>> contours; // ÄÁÅõ¾î 
	vector<Point> approxCurve;
	

};