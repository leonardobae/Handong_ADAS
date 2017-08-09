#include "RM_feature_detect.h"
#include <vector>

using namespace std;
using namespace cv;

RM::RM()
{
}


void RM::subRoiBinary(Mat& img)
{
	Mat sub_roi;

	Scalar aveCheckIntensity, stddev;
	int sub_roi_width = roi.cols / num_sub_roi_x; // roi 하나 작은것의 width 
	int sub_roi_height = roi.rows / num_sub_roi_y; // roi 하나 작은것의 height

	Rect sub_roi_rect(0, 0, sub_roi_width, sub_roi_height);

	sub_roi_rect.x = 0;
	for (int i = 0; i < num_sub_roi_x; i++){
		sub_roi_rect.y = 0;

		for (int j = 0; j < num_sub_roi_y; j++) {

			sub_roi = Mat(roi, sub_roi_rect);  // 총 roi중 세부 쪼갠 부분을 보는 것
			meanStdDev(sub_roi, aveCheckIntensity, stddev);

			inRange(sub_roi, aveCheckIntensity[0] + 1.0 * stddev[0], 255, sub_roi); // 이진화영상 만들기, 2,3번째 인자 사이값만 1로(하얀색,255) 살려서 4번째인자(output)에 이진화 영상 넣음 
			Mat sub_roi_temp(img, sub_roi_rect);
			
			sub_roi.copyTo(sub_roi_temp);
			sub_roi_rect.y += sub_roi_height;

		}
		sub_roi_rect.x += sub_roi_width;
	}
	imshow("??", img);
	//imshow("wht_thres", wht_thres);
}

void RM::widthBinary(Mat& markRoiBinary)
{
	float tau = 10;
	Mat srcGray = roi.clone();//=subRoiBinary.clone();
	//cvtColor(roi, srcGray, CV_BGR2GRAY);
	Mat dstGray(srcGray.rows, srcGray.cols, srcGray.type());
	dstGray.setTo(0);

	int aux = 0;
	for (int j = 0; j < srcGray.rows; ++j)
	{
		unsigned char *ptRowSrc = srcGray.ptr<uchar>(j);
		unsigned char *ptRowDst = dstGray.ptr<uchar>(j);
		tau = 0.1*j + 6;
		if (tau < 0)
			tau = 0;
		if (j == srcGray.rows - 1)
			tau = tau;
		for (int i = tau; i < srcGray.cols - tau; ++i)
		{
			if (ptRowSrc[i] != 0)
			{
				aux = 2 * ptRowSrc[i];
				aux += -ptRowSrc[i - (int)tau];
				aux += -ptRowSrc[i + (int)tau];
				aux += -abs((int)(ptRowSrc[i - (int)tau] - ptRowSrc[i + (int)tau]));

				aux = (aux < 0) ? 0 : (aux);
				aux = (aux > 255) ? 255 : (aux);

				ptRowDst[i] = (unsigned char)aux;

			}
		}
	}

	threshold(dstGray, markRoiBinary, 40, 255, THRESH_BINARY);
	//showImg("MarkBinary", markRoiBinary);
}


//컬러 영상을 binary threshed image로 바꿈.
void RM::imgThresh(Mat& img)
{
	Mat bin_1;
	Mat bin_2;
	Mat temp;
	Mat output;
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	
	cvtColor(img, img, CV_BGR2GRAY);

	GaussianBlur(img, temp, cv::Size(0, 0), 3);
	addWeighted(img, 1.5, temp, -0.5, 0, output);
	morphologyEx(output, output, MORPH_CLOSE, element);
	morphologyEx(output, output, MORPH_OPEN, element);

	output.copyTo(roi);

	threshold(output, output, 163, 255, THRESH_BINARY);
	GaussianBlur(output, output, cv::Size(1, 1), 3);

	output.copyTo(t_frame);
}

void RM::WarpImage(Mat& ROI_src)
{
	vector<Point2f> corners(4);
	corners[0] = Point2f(140, 40);
	corners[1] = Point2f(252, 40);
	corners[2] = Point2f(73, 170);
	corners[3] = Point2f(371, 170);
	Size warpSize(250, 385);
	Mat warpImg(warpSize, ROI_src.type());

	//Warping 후의 좌표
	vector<Point2f> warpCorners(4);
	warpCorners[0] = Point2f(0, 0);
	warpCorners[1] = Point2f(warpImg.cols, 0);
	warpCorners[2] = Point2f(0, warpImg.rows);
	warpCorners[3] = Point2f(warpImg.cols, warpImg.rows);

	//Transformation Matrix 구하기
	Mat trans = getPerspectiveTransform(corners, warpCorners);

	//Warping
	warpPerspective(ROI_src, warpImg, trans, warpSize);

	Cont(warpImg);

	for (int i = 0; i<corners.size(); i++)
		circle(c_frame(Rect(480, 450, 380, 200)), corners[i], 3, Scalar(0, 255, 0), 3);

	imshow("X, Y Points", c_frame(Rect(480, 450, 380, 200)));


}


//Input Param: Color Img
//Output: 
void RM::Cont(Mat& img)
{
	RNG rng(12345);
	Mat output;
	Point pt;

	pt.x = 5; //
	pt.y = 5; //

	imgThresh(img);

	imshow("Thresh", t_frame);

	Mat temp = Mat::zeros(t_frame.rows + 40, t_frame.cols + 40, CV_8UC1);
	t_frame.copyTo(temp(cv::Rect(20, 20, t_frame.cols, t_frame.rows)));
	imshow("temp", temp);
 
	findContours(temp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	cout << "contour size : " << contours.size() << endl;

	vector<Rect> boundRect(contours.size());

	for (size_t i = 0; i < contours.size(); i++)
	{
		
		approxPolyDP(Mat(contours[i]), approxCurve, arcLength(Mat(contours[i]), true)*0.01, true);
		boundRect[i] = boundingRect(Mat(approxCurve));
		
		if (boundRect[i].height > boundRect[i].width && boundRect[i].height < 4 * boundRect[i].width && boundRect[i].height > 100 && boundRect[i].width < 160)
		{ // RoadMark Candidate condition - 임의로 설정한 값임.

			//Mat temp = Mat::zeros(boundRect[i].height + 20, boundRect[i].width + 20, CV_8UC1);
			//Mat temp = Mat::zeros(t_frame.cols + 300, t_frame.rows + 300, CV_8UC1);
			Mat c_box;

			drawContours(temp, vector<vector<Point>>(1, approxCurve), 0, Scalar(255, 255, 255), CV_FILLED);

			boundRect[i].height += 10;
			boundRect[i].width += 10;
			boundRect[i] -= pt;
			//rectangle(t_frame, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 255, 255), 2);
			
			c_box = temp(boundRect[i]);
			resize(c_box, c_box, Size(c_box.cols / 2, c_box.rows / 2));
			candidateBoxes.push_back(c_box);
	
		}
		rectangle(t_frame, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 255, 255), 2);
		
	}
	cout << "candidateBoxes.size() = " << candidateBoxes.size() << endl;
	imshow("Contours", temp);

}

void RM::MyFeatureDetector()
{
	//if (!candidateKeypointVector.empty())			//candiateKeyPointVector 초기화
	//	candidateKeypointVector.clear();

	Mat candidateDescriptors;

	cout << "keypoints.size() " << keypoints.size() << endl;

	// candidate box 의 특징점 검출 
	for (int i = 0; i < candidateBoxes.size(); i++)
	{
		//검출기 생성 
		Mat dstImage = Mat::zeros(candidateBoxes[i].size(), CV_8UC1);
		candidateBoxes[i].copyTo(dstImage);											// <- 두줄 필요??

		//이미 candidateBoxes에 있는 애들이 gray(binary), threshold된 이미지들임.
		//cvtColor(candidateBoxes[i], dstImage, CV_BGR2GRAY);


		//////------------------------GFTTDetector---------------------------//////
		GFTTDetector goodF(maxCorners, qualityLevel, minDistance, blockSize, true);
		goodF.detect(dstImage, keypoints);
		cout << "1. keypoints.size() = " << keypoints.size() << endl;
		candidateKeypointVector.push_back(keypoints);

		//KeyPointsFilter::removeDuplicated(keypoints);
		//KeyPointsFilter::retainBest(keypoints, 10);


		//FREAK freak;
		//freak.compute(dstImage, keypoints, candidateDescriptors);

		//BriefDescriptorExtractor brief(16);
		//brief.compute(dstImage, keypoints, candidateDescriptors);
		//cout << "2. keypoints.size() = " << keypoints.size() << endl;

		//BRISK briskF;
		//briskF.compute(dstImage, keypoints, candidateDescriptors);

		ORB orbF(100, 1.2f, 8, edgeThreshold, 0, 2, ORB::HARRIS_SCORE, patchSize); //임의의 descriptor 설정. small image에 절적한 patch size를 내가 설정할 수 있음... 허나 다른 것들도 설정할수 있을 지 모름 q. 패턴은 또 무엇인가.....
		orbF.compute(dstImage, keypoints, candidateDescriptors);
		cout << "2. keypoints.size() = " << keypoints.size() << endl;
		candidateDescriptorVector.push_back(candidateDescriptors);
		selectedBoxes.push_back(dstImage);
		//////--------------------------------------------------------------//////


		/*
		///////----------------------FAST-------------------------------//////
		FASTX(dstImage, keypoints, 20, true, type);
		KeyPointsFilter::removeDuplicated(keypoints);
		cout << "keypoints.size()" << keypoints.size() << endl;
		//////---------------------------------------------------------//////
		*/

		/*
		//////--------------------------ORB-----------------------------//////
		Mat descriptors;

		//opencv 2410
		//ORB::ORB(100);
		ORB orbF(300);
		orbF.detect(dstImage, keypoints);
		KeyPointsFilter::removeDuplicated(keypoints);
		//KeyPointsFilter::retainBest(keypoints, 20);
		orbF.compute(dstImage, keypoints, descriptors);
		cout << "keypoints.size() = " << keypoints.size() << endl;
		//////---------------------------------------------------------//////
		*/


		KeyPoint element;
		for (int k = 0; k < keypoints.size(); k++)
		{
			element = keypoints[k];

			//cout << element.pt << "," << element.reponse;
			//cout << "," << element.angle ;
			//cout << "," << element.size;
			//cout << "," << element.class_id << endl;

			circle(dstImage, element.pt, cvRound(element.size / 2), Scalar(255), 2);
		}

		namedWindow("dstImage", CV_WINDOW_AUTOSIZE);
		ShowImage("dstImage", dstImage);

		window_open_flag = true;
		no_candidate_cnt = 0;			 //initialize     <- 필요??
		//candidateBoxes.pop_back();      //   <- ????

	}

	cout << "no_candidate_cnt = " << no_candidate_cnt << endl;
	if (candidateBoxes.size() == 0)
	{
		no_candidate_cnt++;
	}

	if (window_open_flag == true && no_candidate_cnt > 10){ // //window가 열려있을 때 10frame 이상 candidate 못찾으면 창 닫음
		window_open_flag = false;
		//candidateDescriptorVector.clear();
		destroyWindow("dstImage");
	}

	return;
}


void RM::SetQueryData()  //쿼리 데이타 구축 
{

	for (int i = 0; i < querySize; i++)
	{
		Mat queryImg = imread("RoadMark" + to_string(i + 1) + ".png", CV_8UC1);

		//Query Contour
		vector<vector<Point>> queryContours;
		findContours(queryImg, queryContours, noArray(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		cout << "queryContour.size() = " << queryContours.size() << endl;
		drawContours(queryImg, queryContours, -1, Scalar(255), 2, 8, hierarchy, 0, Point());

		//cvtColor(queryImg, queryImg, CV_BGR2GRAY);
		resize(queryImg, queryImg, Size(120, 180)); //실험적으로 설정한 값 


		if (queryImg.empty()) {
			cerr << "Image File Open Error" << endl;
			exit(1);
		}

		vector<KeyPoint> queryKeypoints;
		Mat descriptors;

		//////-----------------------GFTTDetector-------------------//////
		GFTTDetector queryGoodF(maxCorners, qualityLevel, minDistance, blockSize, true);
		queryGoodF.detect(queryImg, queryKeypoints);
		cout << "1. queryKeypoints.size()" << queryKeypoints.size() << endl;

		KeyPointsFilter::removeDuplicated(queryKeypoints);
		//KeyPointsFilter::retainBest(queryKeypoints, 20);

		ORB orbF(100, 1.2f, 8, edgeThreshold, 0, 2, ORB::HARRIS_SCORE, patchSize);
		orbF.compute(queryImg, queryKeypoints, descriptors);
		cout << "2. queryKeypoints.size() = " << queryKeypoints.size() << endl;

		//BRISK queryBriskF;
		//queryBriskF.compute(queryImg, queryKeypoints, descriptors);
		//////------------------------------------------------------//////

		/*
		//////-------------------FASTX detector-----------------------//////
		//FAST 특징 검출
		//FASTX(queryImg, queryKeypoints, 10, true, type);
		//KeyPointsFilter::removeDuplicated(queryKeypoints);
		//cout << "queryKeypoints.size()" << queryKeypoints.size() << endl;
		//////------------------------------------------------------//////
		*/

		/*
		//////----------------ORB Detector and Descriptor------------//////
		//opencv 2410 - 문법 정확히 모름...ㅎㅎ;;;
		ORB orbF(100);
		orbF.detect(queryImg, queryKeypoints);
		KeyPointsFilter::removeDuplicated(queryKeypoints);
		KeyPointsFilter::retainBest(queryKeypoints, 20);
		orbF.compute(queryImg, queryKeypoints, descriptors);
		cout << "queryKeypoints.size() = " << queryKeypoints.size() << endl;
		//////------------------------------------------------------//////
		*/

		//특징점 벡터 Keypoints와 기술자 행력 descrptors를 출력한다.
		//FileStorage fs("Keypoints.yml", FileStorage::WRITE);
		//write(fs, "keypoints" + to_string(i + 1), queryKeypoints);
		//write(fs, "descriptors" + to_string(i + 1), descriptors);
		//fs.release();

		//drawKeypoints(queryImg, queryKeypoints, queryImg);

		KeyPoint element;
		for (int k = 0; k < queryKeypoints.size(); k++)
		{
			element = queryKeypoints[k];
			/*
			cout << element.pt << "," << element.reponse;
			cout << "," << element.angle ;
			cout << "," << element.size;
			cout << "," << element.class_id << endl;
			*/
			circle(queryImg, element.pt, cvRound(element.size / 2), Scalar(255), 2);

		}

		DBdescriptorVector.push_back(descriptors);
		DBKeypointVector.push_back(queryKeypoints);
		DBImgVector.push_back(queryImg);
		ShowImage("queryImg" + to_string(i + 1), queryImg);

	}
	return;
}

void RM::ShowImage(String windowName, Mat frame)
{
	if (frame.empty()){
		cerr << "error : no image " << endl;
		exit(1);
	}
	else imshow(windowName, frame);

	return;
}

int RM::DescriptorMatching()
{
	vector<DMatch> finalMatches;
	//1. NORM_MAMMING 매칭 거리로 사용한 Brute Force 매칭 결과 사용.
	for (int j = 0; j < candidateDescriptorVector.size(); j++)
	{
		cout << candidateDescriptorVector.size() << endl;
		int maxMatch_cnt = 0;
		int maxMatch_idx = 0;
		for (int i = 0; i < DBdescriptorVector.size(); i++) //query
		{
			cout << DBdescriptorVector.size() << endl;
			//Matching descriptor vectors
			vector <DMatch> matches;
			BFMatcher matcher(NORM_HAMMING);
			matcher.match(DBdescriptorVector[i], candidateDescriptorVector[j], matches);

			//Ptr<DescriptorMatcher> matcher;
			//matcher = DescriptorMatcher::create("BruteForce-Hamming");
			//matcher->match(descriptorVector[i], candidateDescriptorVector[j], matches);

			cout << "matches.size() = " << matches.size() << endl;
			if (matches.size() < 4)
			{
				cout << "the number of matches is too small" << endl;
				return -1;
			}

			//Find goodmatches such that matches[m].distance <= 4*minDist
			double minDist, maxDist;
			minDist = maxDist = matches[0].distance; //initialize 
			for (int m = 0; m < matches.size(); m++)
			{
				double dist = matches[m].distance;
				if (dist < minDist) minDist = dist;
				if (dist > maxDist) maxDist = dist;

			}
			cout << "minDist =" << (double)minDist << endl;
			cout << "maxDist =" << (double)maxDist << endl;


			vector<DMatch> goodMatches;
			//int goodMatchesNumber = 0;

			double fTh = 4 * minDist;
			for (int k = 0; k < matches.size(); k++)
			{
				if (matches[k].distance <= max(fTh, 0.02))
				{
					goodMatches.push_back(matches[k]);
					//goodMatchesNumber++;
				}

			}

			cout << "goodMatches.size()" << goodMatches.size() << endl;
			if (goodMatches.size() < 4){
				cout << "the number of good Matches is too small" << endl;
				break;
			}

			//Find the best query image matching the candidate boxes
			if (maxMatch_cnt < goodMatches.size())
			{
				maxMatch_cnt = goodMatches.size();
				maxMatch_idx = i;
				if (finalMatches.size() > 0) finalMatches.pop_back();
				finalMatches = goodMatches; // vector<DMatch> 가능한 연산인지 모름...ㅋ
			}


			/*
			//Find homography between keypoints1 and keypoints2
			vector<Point2f> obj;
			vector<Point2f> scene;
			for (int n = 0; n < goodMatches.size(); n++)
			{
			//Get the keypoints from the good matches

			}
			*/

		}
		cout << maxMatch_idx << "th query is best matching" << endl;
		//Draw good_matches;
		Mat imgMatches;
		drawMatches(DBImgVector[maxMatch_idx], DBKeypointVector[maxMatch_idx], selectedBoxes[j], candidateKeypointVector[j], finalMatches, imgMatches,
			Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS); //DEFAULT
		ShowImage("goodMatches", imgMatches);
	}

	//candidateKeypointVector.clear();
	//candidateDescriptorVector.clear();

	return 0;
}

int RM::VectorClear()
{

	cout << "vector clear: no_candidate_cnt  " << no_candidate_cnt << endl;
	if (window_open_flag2 == true && no_candidate_cnt > 10)
	{

		selectedBoxes.clear();
		candidateKeypointVector.clear();
		candidateDescriptorVector.clear();
		cout << selectedBoxes.size() << " & " << candidateKeypointVector.size() << " & " << candidateDescriptorVector.size() << endl;
		window_open_flag2 = false;
		destroyWindow("goodMatches");
		return 1;
	}

	return 0;
}