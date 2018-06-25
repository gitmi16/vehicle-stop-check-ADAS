#include "opencv2/opencv.hpp"
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include <cstdio>

using namespace std;
using namespace cv;

int main() {
	
	// Create a VideoCapture object
	VideoCapture cap("C:/Users/Hi/Desktop/visteon.mp4");

	cap.set(1, 100);	//starting from 100th frame 
	Mat Iref;
	cap >> Iref;	//previous frame
	while (1)
	{
		Mat I,temp,original,distorted;		
		int k = 1;
		while (k % 3 != 0)	//skip 2 frames
		{
			cap >> temp;
			k++;
		}
		cap >> I;	//current frame
		
		// If the Iref or I is empty, break immediately
		if (Iref.empty()|| I.empty())
			break;

		cvtColor(Iref, original, CV_RGB2GRAY);	//converting to grayscale
		cvtColor(I, distorted, CV_RGB2GRAY);	//converting to grayscale

		equalizeHist(original, original);	//histogram equalization
		equalizeHist(distorted, distorted);	//histogram equalization
		
		std::vector<KeyPoint> ptsOriginal, ptsDistorted;
		Mat featuresOriginal, featuresDistorted;
		Ptr<FeatureDetector> detector = ORB::create();	//ORB detector
		Ptr<DescriptorExtractor> descriptor = ORB::create();	//ORB feature extractor
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");	//descriptor matcher
		detector->detect(original, ptsOriginal);	//detect keypoints in previous image
		detector->detect(distorted, ptsDistorted);	//detect keypoints in current image
		descriptor->compute(original, ptsOriginal, featuresOriginal);	//compute feature in previous image
		descriptor->compute(distorted, ptsDistorted, featuresDistorted);	//compute feature in current image
		
		vector<DMatch> index_pairs;
		
		matcher->match(featuresOriginal, featuresDistorted, index_pairs);	//matching prev and current frame features
		
		std::vector< DMatch > good_index_pairs;
		for (int i = 0; i < featuresOriginal.rows; i++)
		{
			if (index_pairs[i].distance <= 30.0)	//threholding distance between matched pairs to get strong matching points
			{
				good_index_pairs.push_back(index_pairs[i]);
			}
		}
		
		vector<Point2f> matchedPtsOriginal, matchedPtsDistorted;
		int j = 1;
			for (int i = 0; i < good_index_pairs.size(); i++)
			{
				if (good_index_pairs[i].distance < 20.0)	//taking only 4 matched points whose distance is less than 20
				{
					matchedPtsOriginal.push_back(ptsOriginal[good_index_pairs[i].queryIdx].pt);
					matchedPtsDistorted.push_back(ptsDistorted[good_index_pairs[i].trainIdx].pt);
					j++;
				}
				if (j > 4)
				{
					break;
				}
			}
		
		Point2f inputQuad[4];
		Point2f outputQuad[4];
		inputQuad[0] = matchedPtsOriginal[0];
		inputQuad[1] = matchedPtsOriginal[1];
		inputQuad[2] = matchedPtsOriginal[2];
		inputQuad[3] = matchedPtsOriginal[3];
		outputQuad[0] = matchedPtsDistorted[0];
		outputQuad[1] = matchedPtsDistorted[1];
		outputQuad[2] = matchedPtsDistorted[2];
		outputQuad[3] = matchedPtsDistorted[3];

		Mat M = getPerspectiveTransform(inputQuad, outputQuad);		//calc perspective transform from 4 points
		double x=M.at<double>(0,2);		
		double y=M.at<double>(1,2);
		double disp = (x*x) + (y*y);		//changes in horizontal and vertical direction
		//char str[500];
		//sprintf_s(str, "%f  disp", disp);
		if (disp < 0.1)		
		{
			//putText(I, str, Point2f(620, 20), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255));
			putText(I, "STOP", Point2f(550, 20), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255));
		}
		imshow("win", I);
		waitKey(1);

		Iref = I;		//taking current frame as prev frame
	}
	 
	cap.release();
	// Closes all the windows
	destroyAllWindows();
	return 0;
}