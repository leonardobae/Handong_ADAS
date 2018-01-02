//#pragma once
//#include "RMD.h"
//
//RMD::RMD()
//{
//}
//
//
//Mat RMD::WarpImage(Mat& ROI_src)
//{
//	vector<Point2f> corners(4);
//	corners[0] = Point2f(150, 40);
//	corners[1] = Point2f(278, 40);
//	corners[2] = Point2f(53, 170);
//	corners[3] = Point2f(365, 170);
//	Size warpSize(250, 385);
//	Mat warpImg(warpSize, ROI_src.type());
//
//	//Warping 후의 좌표
//	vector<Point2f> warpCorners(4);
//	warpCorners[0] = Point2f(0.0f, 0.0f);
//	warpCorners[1] = Point2f(warpImg.cols, 0.0f);
//	warpCorners[2] = Point2f(0.0f, warpImg.rows);
//	warpCorners[3] = Point2f(warpImg.cols, warpImg.rows);
//
//	//Transformation Matrix 구하기
//	Mat trans = getPerspectiveTransform(corners, warpCorners);
//
//	//Warping
//	warpPerspective(ROI_src, warpImg, trans, warpSize);
//
//	//Cont(warpImg);
//
//	for (int i = 0; i<corners.size(); i++)
//		circle(cur_frame(Rect(480, 450, 380, 200)), corners[i], 3, Scalar(0, 0, 255), 3);
//
//	imshow("X, Y Points", cur_frame(Rect(480, 450, 380, 200)));
//
//	return warpImg;
//}