#include "main.h"
#include <opencv2\opencv.hpp>
#include "opencv2/core/core.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include <opencv2/features2d/features2d.hpp>  
#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/legacy/legacy.hpp"   
#include "highgui/highgui.hpp"    
#include <math.h>

using namespace cv;
using namespace std;

main::main()
{
}


main::~main()
{
}

void drawArrow(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha, cv::Scalar& color, int thickness, int lineType)
{
	if (pStart == pEnd)
	{
		//只画一个点
		circle(img, pStart, 1, color);
		return;
	}

	const double PI = 3.1415926;
	Point arrow;
	//计算 θ 角（最简单的一种情况在下面图示中已经展示，关键在于 atan2 函数，详情见下面）   
	double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));

	line(img, pStart, pEnd, color, thickness, lineType);

	//计算箭角边的另一端的端点位置（上面的还是下面的要看箭头的指向，也就是pStart和pEnd的位置） 
	arrow.x = pEnd.x + len * cos(angle + PI * alpha / 180);

	arrow.y = pEnd.y + len * sin(angle + PI * alpha / 180);

	line(img, pEnd, arrow, color, thickness, lineType);

	arrow.x = pEnd.x + len * cos(angle - PI * alpha / 180);

	arrow.y = pEnd.y + len * sin(angle - PI * alpha / 180);

	line(img, pEnd, arrow, color, thickness, lineType);
}


//计算质心
float GetCenterOfMass(Mat m)
{
	float m_00 = 0, m_01 = 0, m_10 = 0;

	for (int x = 0; x < m.rows; x++)
	{
		for (int y = 0; y < m.cols; y++)
		{
			m_00 += m.at<uchar>(x, y);
			m_01 += (float)y * m.at<uchar>(x, y);
			m_10 += (float)x * m.at<uchar>(x, y);
		}
	}

	float x_c = m_10 / m_00;
	float y_c = m_01 / m_00;

	return fastAtan2(m_01, m_10);

	//return Point2f(x_c, y_c);
}

//计算灰度直方图
Mat getHistograph(const Mat grayImage)
{
	//定义求直方图的通道数目，从0开始索引  
	int channels[] = { 0 };
	//定义直方图的在每一维上的大小，例如灰度图直方图的横坐标是图像的灰度值，就一维，bin的个数  
	//如果直方图图像横坐标bin个数为x，纵坐标bin个数为y，则channels[]={1,2}其直方图应该为三维的，Z轴是每个bin上统计的数目  
	const int histSize[] = { 256 };
	//每一维bin的变化范围  
	float range[] = { 0,256 };

	//所有bin的变化范围，个数跟channels应该跟channels一致  
	const float* ranges[] = { range };

	//定义直方图，这里求的是直方图数据  
	Mat hist;
	//opencv中计算直方图的函数，hist大小为256*1，每行存储的统计的该行对应的灰度值的个数  
	calcHist(&grayImage, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);

	//找出直方图统计的个数的最大值，用来作为直方图纵坐标的高  
	double maxValue = 0;
	//找矩阵中最大最小值及对应索引的函数  
	minMaxLoc(hist, 0, &maxValue, 0, 0);
	//最大值取整  
	int rows = cvRound(maxValue);
	//定义直方图图像，直方图纵坐标的高作为行数，列数为256(灰度值的个数)  
	//因为是直方图的图像，所以以黑白两色为区分，白色为直方图的图像  
	Mat histImage = Mat::zeros(rows, 256, CV_8UC1);

	//直方图图像表示  
	for (int i = 0; i<256; i++)
	{
		//取每个bin的数目  
		int temp = (int)(hist.at<float>(i, 0));
		//如果bin数目为0，则说明图像上没有该灰度值，则整列为黑色  
		//如果图像上有该灰度值，则将该列对应个数的像素设为白色  
		if (temp)
		{
			//由于图像坐标是以左上角为原点，所以要进行变换，使直方图图像以左下角为坐标原点  
			histImage.col(i).rowRange(Range(rows - temp, rows)) = 255;
		}
	}
	//由于直方图图像列高可能很高，因此进行图像对列要进行对应的缩减，使直方图图像更直观  
	Mat resizeImage;
	resize(histImage, resizeImage, Size(256, 256));
	return resizeImage;
}


void main()
{
	double start = static_cast<double>(getTickCount());
	Mat src1, gray1, src2, gray2;
	src1 = imread("img_00000.bmp");
	src2 = imread("img_00001.bmp");


	//src1 = imread("Farm_REF.bmp");
	//src2 = imread("Farm_IR.bmp");

	//Mat image = imread("2.png");

	//vector<KeyPoint> keypoints;

	//SiftFeatureDetector surf;
	//surf.detect(image, keypoints);

	//drawKeypoints(image, keypoints, image, Scalar(255, 0, 0),
	//	DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//for (int i = 0; i < keypoints.size(); i++)
	//{
	//	Point start = Point(keypoints[i].pt.x, keypoints[i].pt.y);
	//	Point end;

	//	float angle = keypoints[i].angle;


	//	//int th = 3;
	//	//Rect rect(start.x - th, start.y - th, 2 * th, 2 * th);
	//	//Mat roi = image(rect);
	//	//float angle = GetCenterOfMass(roi);

	//	int len = 20;
	//	int end_x = cos(angle) * len;
	//	int end_y = sin(angle) * len;
	//	end = Point(end_x+ start.x, end_y+ start.y);
	//	drawArrow(image, start, end, 5, 30, Scalar(255 , 0 , 0), 1, 1);
	//}

	//imshow("c",image);
	//imwrite("fang.png", image);

	cvtColor(src1, gray1, CV_BGR2GRAY);
	cvtColor(src2, gray2, CV_BGR2GRAY);

	//Mat hist1 = getHistograph(gray1);
	//Mat hist2 = getHistograph(gray2);

	//imshow("hist1", hist1);
	//imshow("hist2", hist2);

	//imwrite("hist1.png", hist1);
	//imwrite("hist2.png", hist2);

	morphologyEx(gray1, gray1, MORPH_GRADIENT, Mat());
	morphologyEx(gray2, gray2, MORPH_GRADIENT, Mat());

	//imshow("mor1", gray1);
	//imshow("mor2", gray2);
	//imwrite("mor1.png", gray1);
	//imwrite("mor2.png", gray2);

	//hist1 = getHistograph(gray1);
	//hist2 = getHistograph(gray2);

	//imshow("hist3", hist1);
	//imshow("hist4", hist2);
	//imwrite("hist3.png", hist1);
	//imwrite("hist4.png", hist2);

	vector<KeyPoint> keypoints1, keypoints2;
	Mat image1_descriptors, image2_descriptors;

	SurfFeatureDetector detector;
	//BriskFeatureDetector detector;
	//SurfFeatureDetector descriptor;
	BriefDescriptorExtractor descriptor;

	//ORB detector;     //采用ORB算法提取特征点  
	detector.detect(gray1, keypoints1);
	detector.detect(gray2, keypoints2);

	//for (int i = 0; i < keypoints1.size(); i++)
	//{
	//	Point start = Point(keypoints1[i].pt.x, keypoints1[i].pt.y);
	//	int th = 5;
	//	Rect rect(start.x - th, start.y - th, 2 * th, 2 * th);
	//	Mat roi = gray1(rect);
	//	float angle = GetCenterOfMass(roi);
	//	keypoints1[i].angle = angle;
	//}

	//for (int i = 0; i < keypoints2.size(); i++)
	//{
	//	Point start = Point(keypoints2[i].pt.x, keypoints2[i].pt.y);
	//	int th = 5;
	//	Rect rect(start.x - th, start.y - th, 2 * th, 2 * th);
	//	Mat roi = gray2(rect);
	//	float angle = GetCenterOfMass(roi);
	//	keypoints2[i].angle = angle;
	//}

	descriptor.compute(gray1, keypoints1, image1_descriptors);
	descriptor.compute(gray2, keypoints2, image2_descriptors);

	BFMatcher matcher(NORM_HAMMING, true); //汉明距离做为相似度度量  
	//BruteForceMatcher<L2<float>> matcher;

	vector<DMatch> matches;
	matcher.match(image1_descriptors, image2_descriptors, matches);

	sort(matches.begin(), matches.end());

	Mat match_img;
	//drawMatches(src1, keypoints1, src2, keypoints2, matches, match_img);
	//imshow("滤除误匹配前", match_img);

	//保存匹配对序号  
	vector<int> queryIdxs(matches.size()), trainIdxs(matches.size());
	for (size_t i = 0; i < matches.size(); i++)
	{
		queryIdxs[i] = matches[i].queryIdx;
		trainIdxs[i] = matches[i].trainIdx;
	}

	Mat H12;   //变换矩阵  

	vector<Point2f> points1; KeyPoint::convert(keypoints1, points1, queryIdxs);
	vector<Point2f> points2; KeyPoint::convert(keypoints2, points2, trainIdxs);
	int ransacReprojThreshold = 5;  //拒绝阈值  
	H12 = findHomography(Mat(points1), Mat(points2), CV_RANSAC, ransacReprojThreshold);
	vector<char> matchesMask(matches.size(), 0);
	Mat points1t;
	perspectiveTransform(Mat(points1), points1t, H12);

	int mask_sum = 0;

	for (size_t i1 = 0; i1 < points1.size(); i1++)  //保存‘内点’  
	{
		if (norm(points2[i1] - points1t.at<Point2f>((int)i1, 0)) <= ransacReprojThreshold) //给内点做标记  
		{
			matchesMask[i1] = 1;
			mask_sum++;
		}
	}


	Mat Mat_img;
	drawMatches(src1, keypoints1, src2, keypoints2, matches, Mat_img, Scalar(0, 0, 255), Scalar::all(-1), matchesMask);

	imshow("ransac筛选后", Mat_img);

	imwrite("result.png", Mat_img);

	double time = ((double)getTickCount() - start) / getTickFrequency();
	cout << "所用时间为：" << time << "秒" << endl;

	cout << "图1找到特征点：" << keypoints1.size() << endl;
	cout << "图2找到特征点：" << keypoints2.size() << endl;
	cout << "一共找到匹配点对：" << matches.size() << endl;
	cout << "正确匹配点对：" << mask_sum << endl;

	waitKey(0);

}