//------------------------------------------------------------------------------
// AnonymousKinect_Application.cpp
// 
// University of Florida
// Francesco Pittaluga
// f.pittaluga@ufl.edu
//------------------------------------------------------------------------------

#pragma comment(lib, "user32.lib")

#include "stdafx.h"
#include <strsafe.h>
#include <math.h>
#include <limits>
#include <Wincodec.h>
#include "iostream"
#include <windows.h>
#include <algorithm>
#include "App_predict.h"
#include <time.h>
#include <fstream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#ifndef HINST_THISCOMPONENT
EXTERN_C IMAGE_DOS_HEADER __ImageBase;
#define HINST_THISCOMPONENT ((HINSTANCE)&__ImageBase)
#endif

# define M_PI           3.14159265358979323846  /* pi */
#define FRAME_RANGE 9 // Number of frames to be captured
#define FRAME_INC 1
#define VEC_INC 3980   // Size of feature vector for single frame
#define KEYDOWN(vkcode) (GetAsyncKeyState(vkcode) & 0x8000 ? vkcode : 0)
#define VK_1 0x31
#define VK_2 0x32
#define VK_3 0x33
#define VK_4 0x34
#define VK_5 0x35
#define VK_6 0x36
#define VK_7 0x37
#define VK_D 0x44
#define VK_S 0x53

// Demo global variables
bool switchView = 0;
cv::Mat mAnonymizingFace;
bool DBMatch = true;
bool blackOutBody = true;
int slide = 2;
int frameIter;
int frameNum[6] = { 0, 0, 0, 0, 0, 0 };
int vec_inc[6] = { 0 };
int incs[6] = { 0 };
std::ofstream myfile;
std::vector<std::vector<float>> skelVec;
std::vector<std::vector<float>> depthVec;
std::vector<std::vector<float>> featVec;

// Empty body vector
bool trackFlag = false;


cv::Mat trainData;
cv::Mat labels;
cv::Ptr<cv::ml::SVM> svm;

using namespace std;
using namespace cv;

/// <summary>
/// Constructor
/// </summary>
App_predict::App_predict(std::string mainWinName, std::string sideWinName, std::string dispWinName, std::string dispCallibFileName, std::string infraredCallibFileName) :
m_pKinectSensor(NULL),
m_pCoordinateMapper(NULL),
m_pMultiSourceFrameReader(NULL),
m_pDepthCoordinates(NULL),
m_pColorRGBX(NULL),
m_pInfraredRGBX(NULL),
cDisplayCallibFileName(dispCallibFileName),
cInfraredCallibFileName(infraredCallibFileName),
cMainWinName(mainWinName),
cDisplayWinName(dispWinName),
cSecondWinName(sideWinName),
flag(false)
{

	// create heap storage for infrared pixel data in RGBX format
	m_pInfraredRGBX = new RGBQUAD[cInfraredWidth * cInfraredHeight];

	// create heap storage for color pixel data in RGBX format
	m_pColorRGBX = new RGBQUAD[cColorWidth * cColorHeight];

	// create heap storage for the coorinate mapping from color to depth
	m_pDepthCoordinates = new DepthSpacePoint[cColorWidth * cColorHeight];

	// create heap storage for the coorinate mapping from depth to color
	m_pColorCoordinates = new ColorSpacePoint[cColorWidth * cColorHeight];
}


/// <summary>
/// Destructor
/// </summary>
App_predict::~App_predict()
{
	if (m_pColorRGBX)
	{
		delete[] m_pColorRGBX;
		m_pColorRGBX = NULL;
	}

	if (m_pDepthCoordinates)
	{
		delete[] m_pDepthCoordinates;
		m_pDepthCoordinates = NULL;
	}

	// done with frame reader
	SafeRelease(m_pMultiSourceFrameReader);

	// done with coordinate mapper
	SafeRelease(m_pCoordinateMapper);

	// close the Kinect Sensor
	if (m_pKinectSensor)
	{
		m_pKinectSensor->Close();
	}

	SafeRelease(m_pKinectSensor);
}

/// <summary>
/// Creates the main window and begins processing
/// </summary>
/// <param name="hInstance">handle to the application instance</param>
/// <param name="nCmdShow">whether to display minimized, maximized, or normally</param>
std::vector<float> App_predict::Run()
{
	// Initialize kinect sensor, callibrated coordinate mapper, and face db
	InitializeDefaultSensor();
	//InitCallibCoordinateMapper();
	featVec.clear();

	for (int i = 0; i < 6; i++) {
		std::vector<float> temp;
		featVec.push_back(temp);
	}
	svm = cv::ml::SVM::load<cv::ml::SVM>("SVM.xml");
	// Output console Message
	char menutext[] = "----------------------------------------------------------------------------\n"
		"*** Data Recording and Training Menu (Press # to select option) *** \n"
		"----------------------------------------------------------------------------\n"
		"(1) Go Back to Main Menu.\n"
		"(2) Press S to start/stop Recording.\n"
		"(3) Press D to switch view \n";
	std::cout << menutext;
		
	// Main program loop
	while (!keyWasPressed(VK_1))
	{
		if (KEYDOWN(VK_S)){

			while (!keyWasPressed(VK_S)){			
				Update();
				if (KEYDOWN(VK_D)){
					if (switchView) switchView = 0;
					else switchView = 1;
					Sleep(500);
				}

			}
			Sleep(500);	
			system("cls");
		}
	}
	trainData.release();
	labels.release();
	return featVec[0];
}

float angleBetween(const cv::Point2f &v1, const cv::Point2f &v2)
{
	float len1 = sqrt(v1.x * v1.x + v1.y * v1.y);
	float len2 = sqrt(v2.x * v2.x + v2.y * v2.y);

	float dot = v1.x * v2.x + v1.y * v2.y;

	float a = dot / (len1 * len2);

	if (a >= 1.0)
		return 0.0;
	else if (a <= -1.0)
		return M_PI;
	else
		return acos(a); // 0..PI
}


cv::Mat rotateMat(cv::Mat tobeRotated, float angle)
{
	cv::Point2f src_center(tobeRotated.cols / 2.0F, tobeRotated.rows / 2.0F);
	cv::Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
	cv::Mat dst;
	warpAffine(tobeRotated, dst, rot_mat, tobeRotated.size());
	return dst;
}



void DrawPoints(std::vector<cv::Point2f>* points, cv::Mat frame, int radius, cv::Scalar color) {
	for (int i = 0; i < (*points).size(); i++)
		cv::circle(frame, cvPoint((*points)[i].x, (*points)[i].y), radius, color, -1, 8, 0);
}


bool yComp(cv::Point2f pt1, cv::Point2f pt2)
{
	return (pt1.y < pt2.y);
}

bool xComp(cv::Point2f pt1, cv::Point2f pt2)
{
	return (pt1.x < pt2.x);
}

/// <summary>
/// Draws quadrilateral from 4 points
/// </summary>
/// <returns>indicates success or failure</returns>
void DrawQuadrilateral(cv::Mat frame, cv::Point2f topLeft, cv::Point2f bottomLeft, cv::Point2f topRight, cv::Point2f BotttomRight, cv::Scalar color, int thickness)
{
	cv::line(frame, topLeft, topRight, color, thickness);
	cv::line(frame, topLeft, bottomLeft, color, thickness);
	cv::line(frame, BotttomRight, topRight, color, thickness);
	cv::line(frame, BotttomRight, bottomLeft, color, thickness);
}

/// <summary>
/// converts rgb quad to mat
/// </summary>
void rgbquad2RGBMat(RGBQUAD* r, cv::Mat m)
{
	RGBQUAD* pSrc = r;

	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			m.at<cv::Vec3b>(i, j)[0] = pSrc->rgbBlue;
			m.at<cv::Vec3b>(i, j)[1] = pSrc->rgbGreen;
			m.at<cv::Vec3b>(i, j)[2] = pSrc->rgbRed;
			++pSrc;
		}
	}
}


Mat imagesc(Mat map, bool cm)
{

	Mat colorMap;
	Mat normMap;
	normalize(map, normMap, 0, 255, NORM_MINMAX, CV_8UC1);
	if (cm) applyColorMap(normMap, colorMap, COLORMAP_JET);
	else colorMap = normMap;


	return colorMap;
}




double maxInfrared = 0;
double maxDepth = 0;


void App_predict::LogExecutionData()
{
	//Create experiment directorie
	time_t t = time(0);
	struct tm now;
	localtime_s(&now, &t);
	char expDir[50];

	sprintf_s(expDir, "Experiments\\%04d%02d%02d_%02d%02d%02d",
		(now.tm_year + 1900), (now.tm_mon + 1), now.tm_mday,
		now.tm_hour, now.tm_min, now.tm_sec);
	CreateDirectoryA(expDir, NULL);

	//save output data
	char outputDir[50];

	//create ouput dir
	sprintf_s(outputDir, "%s\\Output", expDir);
	CreateDirectoryA(outputDir, NULL);

	for (int i = 0; i < m_vOutputFrames.size(); i++) {
		char frName[60];
		sprintf_s(frName, "%s\\%05d.jpg", outputDir, i);
		cv::imwrite(frName, m_vOutputFrames.at(i));
	}

	//save infrared data
	char infraredDir[50];

	//create infrared dir
	sprintf_s(infraredDir, "%s\\Infrared", expDir);
	CreateDirectoryA(infraredDir, NULL);

	for (int i = 0; i < m_vInfraredFrames.size(); i++) {
		char frName[60];
		sprintf_s(frName, "%s\\%05d.jpg", infraredDir, i);
		cv::imwrite(frName, m_vInfraredFrames.at(i));
	}
}


/// <summary>
/// Handle new depth and color data
/// <param name="nTime">timestamp of frame</param>
/// <param name="pDepthBuffer">pointer to depth frame data</param>
/// <param name="nDepthWidth">width (in pixels) of input depth image data</param>
/// <param name="nDepthHeight">height (in pixels) of input depth image data</param>
/// <param name="pColorBuffer">pointer to color frame data</param>
/// <param name="nColorWidth">width (in pixels) of input color image data</param>
/// <param name="nColorHeight">height (in pixels) of input color image data</param>
/// <param name="pBodyIndexBuffer">pointer to body index frame data</param>
/// <param name="nBodyIndexWidth">width (in pixels) of input body index data</param>
/// <param name="nBodyIndexHeight">height (in pixels) of input body index data</param>
/// </summary>
void App_predict::ProcessFrame(
	UINT16* pDepthBuffer, int nDepthWidth, int nDepthHeight,
	UINT16* pInfraredBuffer, int nInfraredWidth, int nInfraredHeight,
	RGBQUAD* pColorBuffer, int nColorWidth, int nColorHeight,
	BYTE* pBodyIndexBuffer, int nBodyIndexWidth, int nBodyIndexHeight,
	int nBodyCount, IBody** ppBodies)
{
	// debug
	bool mainDisp = true;
	bool secDisp = true;
	bool foundBody = false;
	int body_count = 0;
	// Empty skeleton vectors
	vector<float> empty_bc5(1020, 0.0); vector<float> empty_bc4(816, 0.0);
	vector<float> empty_bc3(612, 0.0); vector<float> empty_bc2(408, 0.0); vector<float> empty_bc1(204, 0.0);

	// make sure we've received valid data
	if (m_pCoordinateMapper && m_pDepthCoordinates &&
		pDepthBuffer && (nDepthWidth == cDepthWidth) && (nDepthHeight == cDepthHeight) &&
		pColorBuffer && (nColorWidth == cColorWidth) && (nColorHeight == cColorHeight) &&
		pBodyIndexBuffer && (nBodyIndexWidth == cDepthWidth) && (nBodyIndexHeight == cDepthHeight))
	{
		Mat mainFrame;
		Mat secFrame;

		// Convert raw frames from kinect to opencv mat
		cv::Mat depthFrame = cv::Mat(cv::Size(nDepthWidth, nDepthHeight), CV_16UC1, pDepthBuffer, cv::Mat::AUTO_STEP);
		cv::Mat infraredFrame = cv::Mat(cv::Size(nInfraredWidth, nInfraredHeight), CV_16UC1, pInfraredBuffer, cv::Mat::AUTO_STEP);
		cv::Mat bodyFrame = cv::Mat(cv::Size(nBodyIndexWidth, nBodyIndexHeight), CV_8UC1, pBodyIndexBuffer, cv::Mat::AUTO_STEP);
		cv::Mat colorFrame(cv::Size(nColorWidth, nColorHeight), CV_8UC4, reinterpret_cast<void*>(pColorBuffer));


		cv::Mat scDepthFrame;
		cv::Mat mask = cv::Mat::zeros(424, 512, CV_8UC1); // all 0
		//cv::Mat mask2 = cv::Mat::zeros(424, 512, CV_8UC1); // all 0
		//imagesc for depth image
		double min;
		double max;
		if (frameIter % 1000 == 0)
		{
			cv::Mat temp;
			medianBlur(depthFrame, temp, 5);
			cv::minMaxIdx(temp, &min, &max);
			maxDepth = max;
		}
		frameIter++;

		cv::convertScaleAbs(depthFrame, scDepthFrame, 255 / maxDepth);
		cv::Mat hog_im;
		cv::Mat scBodyFrame;
		cv::resize(scDepthFrame, hog_im, cv::Size(64, 128));
		cv::resize(bodyFrame, scBodyFrame, cv::Size(64, 128));

		//imagesc for depth image
		cv::minMaxIdx(infraredFrame, &min, &max);
		cv::Mat scInfraredFrame;
		if (frameNum == 0)
		{
			cv::Mat temp;
			medianBlur(infraredFrame, temp, 5);
			cv::minMaxIdx(temp, &min, &max);
			maxInfrared = max;
		}
		cv::convertScaleAbs(infraredFrame, scInfraredFrame, 255 / maxInfrared);



		// get joints joints
		std::vector<cv::Point2f> pBodyJoints[BODY_COUNT];
		//cv::Mat mBodyJoints = cv::Mat(cv::Size(nBodyIndexWidth, nBodyIndexHeight), CV_8UC1, pBodyIndexBuffer, cv::Mat::AUTO_STEP);
		for (int i = 0; i < nBodyCount; i++)
		{

			IBody* pBody = ppBodies[i];
			if (pBody)
			{
				BOOLEAN bTracked = false;
				HRESULT hr = pBody->get_IsTracked(&bTracked);
				if (SUCCEEDED(hr) && bTracked)
				{

					//cout << body_count << endl;
					//body_count++;
					flag = true;
					Joint joints[JointType_Count];
					JointOrientation joints_or[JointType_Count];
					hr = pBody->GetJoints(_countof(joints), joints);
					hr = pBody->GetJointOrientations(_countof(joints_or), joints_or);
					if (SUCCEEDED(hr))
					{
						body_count++;
						std::vector<cv::Point2f> vJoints;
						//myfile << frameNum ;
						int temp1[25] = { 511 };
						int temp2[25] = { 0 };

						for (int j = 0; j < _countof(joints); j++)
						{
							DepthSpacePoint depthPoint = { 0 };
							m_pCoordinateMapper->MapCameraPointToDepthSpace(joints[j].Position, &depthPoint);
							vJoints.push_back(cv::Point2f(depthPoint.X, depthPoint.Y));

							std::vector<float> temp_vector_joints;

							featVec[i].push_back(joints[j].Position.X);
							featVec[i].push_back(joints[j].Position.Y);
							featVec[i].push_back(joints[j].Position.Z);
							featVec[i].push_back(joints[j].TrackingState);
							featVec[i].push_back(joints_or[j].Orientation.w);
							featVec[i].push_back(joints_or[j].Orientation.x);
							featVec[i].push_back(joints_or[j].Orientation.y);
							featVec[i].push_back(joints_or[j].Orientation.z);

							//featVec[i].push_back(temp_vector_joints);



						}

						pBodyJoints[i] = vJoints;


					}
				}
			}
		}



		applyColorMap(scDepthFrame, scDepthFrame, cv::COLORMAP_JET);

		//parameters for joint drawing
		cv::Scalar lineColor = cv::Scalar(0, 0, 0);
		int thickness = 3;
		cv::Scalar jointColor = cv::Scalar(255, 255, 255);
		int radius = 3;
		bool foundBodies = false;

		if (switchView) {
			secFrame = scDepthFrame;
			cv::cvtColor(infraredFrame * 10, mainFrame, CV_GRAY2RGB);
			//mainFrame = infraredFrame * 10;
		}
		else {
			mainFrame = scDepthFrame;
			secFrame = infraredFrame * 10;
		}


		for (int j = 0; j < BODY_COUNT; j++)
		{
			if (!pBodyJoints[j].empty())
			{
				foundBodies = true;
				int x_low = 511;
				int x_high = 0;


				for (int k = 0; k < 25; k++)
				{
					int x_test = static_cast<int>(pBodyJoints[j][k].x + 0.5f);
					if (x_test > -1 && x_test < 511) {
						if (x_test < x_low) {
							x_low = x_test;
						}
						if (x_test > x_high) {
							x_high = x_test;
						}
					}
				}

				std::vector<float> features;
				cv::HOGDescriptor hogdis;
				cv::Mat scMask;
				cv::Mat prod = cv::Mat::zeros(64, 128, CV_8UC1);
				cv::Mat prod2 = cv::Mat::zeros(64, 128, CV_8UC1);
				mask = cv::Mat::zeros(424, 512, CV_8UC1); // all 0
				//mask2 = cv::Mat::zeros(424, 512, CV_8UC1); // all 0

				//if (body_count == 1)
				mask(Rect(std::max(x_low, 0), 0, std::min(x_high - x_low, 511), 423)) = 1;
				/*if (body_count == 2)
				mask2(Rect(std::max(x_low, 0), 0, std::min(x_high - x_low, 511), 423)) = 255;*/
				cv::resize(mask, scMask, cv::Size(64, 128));
				prod = hog_im.mul(scMask) * 0;
				hogdis.compute(prod, features);
				/*if (body_count == 2)
				{
				prod2 = hog_im.mul(scMask);
				hogdis.compute(prod2, features);
				}*/



				featVec[j].insert(std::end(featVec[j]), std::begin(features), std::end(features));


				if (frameNum[j] >= FRAME_RANGE)
				{
					std::vector<float> trainData(featVec[j].begin() + vec_inc[j], featVec[j].end());
					int col = trainData.size();
					//convert feature vect to Mat for
					cv::Mat featMat = cv::Mat(1, col, CV_32FC1);
					std::memcpy(featMat.data, trainData.data(), trainData.size()*sizeof(float));
					int response = SVMachine::Predict(featMat, svm);
					string outp_string;
					double r;
					switch (response)
					{
					case 77:
						//outp_string = "Person %d: Waving";
						r = std::max(40.0, double(pBodyJoints[j][JointType_WristRight].y - pBodyJoints[j][JointType_HandTipRight].y));
						ellipse(mainFrame, pBodyJoints[j][JointType_HandRight], cv::Size(r, r), 0, 0, 360, cv::Scalar(0, 0, 0), 8, 8, 0);
						break;
					default:
						break;
						//outp_string = "Person %d: Not Waving";
						//cout << "Other Activities" << endl;
					}
					//char outp_chars[200];
					//sprintf_s(outp_chars, outp_string.c_str(), j);
					//cv::putText(scDepthFrame, outp_chars, Point(5, 15 + j * 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 2);


					vec_inc[j] += VEC_INC;
				}
				//incs[j] += FRAME_INC;
				frameNum[j]++;



				// Torso
				cv::line(mainFrame, pBodyJoints[j][JointType_Head], pBodyJoints[j][JointType_Neck], lineColor, thickness);
				cv::line(mainFrame, pBodyJoints[j][JointType_Neck], pBodyJoints[j][JointType_SpineShoulder], lineColor, thickness);
				cv::line(mainFrame, pBodyJoints[j][JointType_SpineShoulder], pBodyJoints[j][JointType_SpineMid], lineColor, thickness);
				cv::line(mainFrame, pBodyJoints[j][JointType_SpineMid], pBodyJoints[j][JointType_SpineBase], lineColor, thickness);
				cv::line(mainFrame, pBodyJoints[j][JointType_SpineShoulder], pBodyJoints[j][JointType_ShoulderRight], lineColor, thickness);
				cv::line(mainFrame, pBodyJoints[j][JointType_SpineShoulder], pBodyJoints[j][JointType_ShoulderLeft], lineColor, thickness);
				cv::line(mainFrame, pBodyJoints[j][JointType_SpineBase], pBodyJoints[j][JointType_HipRight], lineColor, thickness);
				cv::line(mainFrame, pBodyJoints[j][JointType_SpineBase], pBodyJoints[j][JointType_HipLeft], lineColor, thickness);

				cv::circle(mainFrame, pBodyJoints[j][JointType_Head], radius, jointColor, -1, 8, 0);
				cv::circle(mainFrame, pBodyJoints[j][JointType_Neck], radius, jointColor, -1, 8, 0);
				cv::circle(mainFrame, pBodyJoints[j][JointType_SpineShoulder], radius, jointColor, -1, 8, 0);
				cv::circle(mainFrame, pBodyJoints[j][JointType_SpineMid], radius, jointColor, -1, 8, 0);
				cv::circle(mainFrame, pBodyJoints[j][JointType_SpineBase], radius, jointColor, -1, 8, 0);
				cv::circle(mainFrame, pBodyJoints[j][JointType_SpineShoulder], radius, jointColor, -1, 8, 0);
				cv::circle(mainFrame, pBodyJoints[j][JointType_ShoulderRight], radius, jointColor, -1, 8, 0);
				cv::circle(mainFrame, pBodyJoints[j][JointType_ShoulderLeft], radius, jointColor, -1, 8, 0);
				cv::circle(mainFrame, pBodyJoints[j][JointType_HipRight], radius, jointColor, -1, 8, 0);
				cv::circle(mainFrame, pBodyJoints[j][JointType_HipLeft], radius, jointColor, -1, 8, 0);

				//// Right Arm
				cv::line(mainFrame, pBodyJoints[j][JointType_ShoulderRight], pBodyJoints[j][JointType_ElbowRight], lineColor, thickness);
				cv::line(mainFrame, pBodyJoints[j][JointType_ElbowRight], pBodyJoints[j][JointType_WristRight], lineColor, thickness);
				cv::line(mainFrame, pBodyJoints[j][JointType_WristRight], pBodyJoints[j][JointType_HandRight], lineColor, thickness);
				cv::line(mainFrame, pBodyJoints[j][JointType_HandRight], pBodyJoints[j][JointType_HandTipRight], lineColor, thickness);
				cv::line(mainFrame, pBodyJoints[j][JointType_WristRight], pBodyJoints[j][JointType_ThumbRight], lineColor, thickness);

				cv::circle(mainFrame, pBodyJoints[j][JointType_ShoulderRight], radius, jointColor, -1, 8, 0);
				cv::circle(mainFrame, pBodyJoints[j][JointType_ElbowRight], radius, jointColor, -1, 8, 0);
				cv::circle(mainFrame, pBodyJoints[j][JointType_WristRight], radius, jointColor, -1, 8, 0);
				cv::circle(mainFrame, pBodyJoints[j][JointType_HandRight], radius, jointColor, -1, 8, 0);


				//// Left Arm
				cv::line(mainFrame, pBodyJoints[j][JointType_ShoulderLeft], pBodyJoints[j][JointType_ElbowLeft], lineColor, thickness);
				cv::line(mainFrame, pBodyJoints[j][JointType_ElbowLeft], pBodyJoints[j][JointType_WristLeft], lineColor, thickness);
				cv::line(mainFrame, pBodyJoints[j][JointType_WristLeft], pBodyJoints[j][JointType_HandLeft], lineColor, thickness);
				cv::line(mainFrame, pBodyJoints[j][JointType_HandLeft], pBodyJoints[j][JointType_HandTipLeft], lineColor, thickness);
				cv::line(mainFrame, pBodyJoints[j][JointType_WristLeft], pBodyJoints[j][JointType_ThumbLeft], lineColor, thickness);



				cv::circle(mainFrame, pBodyJoints[j][JointType_ShoulderLeft], radius, jointColor, -1, 8, 0);
				cv::circle(mainFrame, pBodyJoints[j][JointType_ElbowLeft], radius, jointColor, -1, 8, 0);
				cv::circle(mainFrame, pBodyJoints[j][JointType_WristLeft], radius, jointColor, -1, 8, 0);
				cv::circle(mainFrame, pBodyJoints[j][JointType_HandLeft], radius, jointColor, -1, 8, 0);

				//// Right Leg
				cv::line(mainFrame, pBodyJoints[j][JointType_HipRight], pBodyJoints[j][JointType_KneeRight], lineColor, thickness);
				cv::line(mainFrame, pBodyJoints[j][JointType_KneeRight], pBodyJoints[j][JointType_AnkleRight], lineColor, thickness);
				cv::line(mainFrame, pBodyJoints[j][JointType_AnkleRight], pBodyJoints[j][JointType_FootRight], lineColor, thickness);

				cv::circle(mainFrame, pBodyJoints[j][JointType_HipRight], radius, jointColor, -1, 8, 0);
				cv::circle(mainFrame, pBodyJoints[j][JointType_KneeRight], radius, jointColor, -1, 8, 0);

				//// Left Leg
				cv::line(mainFrame, pBodyJoints[j][JointType_HipLeft], pBodyJoints[j][JointType_KneeLeft], lineColor, thickness);
				cv::line(mainFrame, pBodyJoints[j][JointType_KneeLeft], pBodyJoints[j][JointType_AnkleLeft], lineColor, thickness);
				cv::line(mainFrame, pBodyJoints[j][JointType_AnkleLeft], pBodyJoints[j][JointType_FootLeft], lineColor, thickness);

				cv::circle(mainFrame, pBodyJoints[j][JointType_HipLeft], radius, jointColor, -1, 8, 0);
				cv::circle(mainFrame, pBodyJoints[j][JointType_KneeLeft], radius, jointColor, -1, 8, 0);




			}


		}

		cv::imshow(cSecondWinName, secFrame);
		cv::imshow(cMainWinName, mainFrame);

		//*** Data Save only for frames where skeleton is tracked***
		if (flag == true)
		{

			if (frameNum == 0)
			{
				time_t t = time(0);
				struct tm now;
				localtime_s(&now, &t);
				// Create experiment directory
				sprintf_s(expDir, "Experiments\\%04d%02d%02d_%02d%02d%02d",
					(now.tm_year + 1900), (now.tm_mon + 1), now.tm_mday,
					now.tm_hour, now.tm_min, now.tm_sec);
				CreateDirectoryA(expDir, NULL);

				//create ouput dir
				sprintf_s(depthDir, "%s\\Depth", expDir);
				CreateDirectoryA(depthDir, NULL);

				//create infrared dir
				sprintf_s(infraredDir, "%s\\Infrared", expDir);
				CreateDirectoryA(infraredDir, NULL);

				//create color dir
				sprintf_s(colorDir, "%s\\Color", expDir);
				CreateDirectoryA(colorDir, NULL);

				//create bodyIndex dir
				sprintf_s(bodyDir, "%s\\Body", expDir);
				CreateDirectoryA(bodyDir, NULL);

				//create body dir
				sprintf_s(bodyIndexDir, "%s\\BodyIndex", expDir);
				CreateDirectoryA(bodyIndexDir, NULL);


			}

			//save depth data
			char depthName[60];
			sprintf_s(depthName, "%s\\%015d.tiff", depthDir, frameNum);
			cv::imwrite(depthName, depthFrame);

			//save infrared data
			char infraredName[60];
			sprintf_s(infraredName, "%s\\%015d.tiff", infraredDir, frameNum);
			cv::imwrite(infraredName, infraredFrame);

			//save body index data
			char bodyIndexName[60];
			sprintf_s(bodyIndexName, "%s\\%015d.tiff", bodyIndexDir, frameNum);
			cv::imwrite(bodyIndexName, bodyFrame);

			//save color data
			char colorName[60];
			sprintf_s(colorName, "%s\\%010d.tiff", colorDir, frameNum);
			cv::imwrite(colorName, colorFrame);

			//save body data
			//char bodyName[60];
			//sprintf_s(bodyName, "%s\\%015d.xml", bodyDir, frameNum);
			//cv::FileStorage fs(bodyName, cv::FileStorage::WRITE);
			//if (fs.isOpened())
			//{
			//	fs << "pBodyJoints" << std::vector<std::vector<cv::Point2f>>(pBodyJoints, pBodyJoints + sizeof pBodyJoints / sizeof pBodyJoints[0]);
			//	fs.release();
			//}
			//else
			//{
			//	std::cout << "Error: could not save the body data" << std::endl;
			//}
		}
		//if (flag == true)
		//frameNum++;
		//if(body_count == 2)
		//	sk2_count++;
		//if (body_count == 3)
		//	sk3_count++;
		//flag = false;
		cv::waitKey(100);
	}
}


/// <summary>
/// Main processing function
/// </summary>
void App_predict::Update()
{

	if (!m_pMultiSourceFrameReader)
	{
		return;
	}

	IMultiSourceFrame* pMultiSourceFrame = NULL;
	IDepthFrame* pDepthFrame = NULL;
	IInfraredFrame* pInfraredFrame = NULL;
	IColorFrame* pColorFrame = NULL;
	IBodyIndexFrame* pBodyIndexFrame = NULL;
	IBodyFrame* pBodyFrame = NULL;


	HRESULT hr = m_pMultiSourceFrameReader->AcquireLatestFrame(&pMultiSourceFrame);

	if (SUCCEEDED(hr))
	{
		IDepthFrameReference* pDepthFrameReference = NULL;

		hr = pMultiSourceFrame->get_DepthFrameReference(&pDepthFrameReference);
		if (SUCCEEDED(hr))
		{
			hr = pDepthFrameReference->AcquireFrame(&pDepthFrame);
		}

		SafeRelease(pDepthFrameReference);
	}
	if (SUCCEEDED(hr))
	{
		IInfraredFrameReference* pInfraredFrameReference = NULL;

		hr = pMultiSourceFrame->get_InfraredFrameReference(&pInfraredFrameReference);
		if (SUCCEEDED(hr))
		{
			hr = pInfraredFrameReference->AcquireFrame(&pInfraredFrame);
		}

		SafeRelease(pInfraredFrameReference);
	}
	if (SUCCEEDED(hr))
	{
		IColorFrameReference* pColorFrameReference = NULL;

		hr = pMultiSourceFrame->get_ColorFrameReference(&pColorFrameReference);
		if (SUCCEEDED(hr))
		{
			hr = pColorFrameReference->AcquireFrame(&pColorFrame);
		}

		SafeRelease(pColorFrameReference);
	}

	if (SUCCEEDED(hr))
	{
		IBodyIndexFrameReference* pBodyIndexFrameReference = NULL;

		hr = pMultiSourceFrame->get_BodyIndexFrameReference(&pBodyIndexFrameReference);
		if (SUCCEEDED(hr))
		{
			hr = pBodyIndexFrameReference->AcquireFrame(&pBodyIndexFrame);
		}

		SafeRelease(pBodyIndexFrameReference);
	}

	if (SUCCEEDED(hr))
	{
		IBodyFrameReference* pBodyFrameReference = NULL;

		hr = pMultiSourceFrame->get_BodyFrameReference(&pBodyFrameReference);
		if (SUCCEEDED(hr))
		{
			hr = pBodyFrameReference->AcquireFrame(&pBodyFrame);
		}

		SafeRelease(pBodyFrameReference);
	}

	if (SUCCEEDED(hr))
	{
		IFrameDescription* pDepthFrameDescription = NULL;
		int nDepthWidth = 0;
		int nDepthHeight = 0;
		UINT nDepthBufferSize = 0;
		UINT16 *pDepthBuffer = NULL;

		IFrameDescription* pInfraredFrameDescription = NULL;
		int nInfraredWidth = 0;
		int nInfraredHeight = 0;
		UINT nInfraredBufferSize = 0;
		UINT16 *pInfraredBuffer = NULL;

		IFrameDescription* pColorFrameDescription = NULL;
		int nColorWidth = 0;
		int nColorHeight = 0;
		ColorImageFormat imageFormat = ColorImageFormat_None;
		UINT nColorBufferSize = 0;
		RGBQUAD *pColorBuffer = NULL;

		IFrameDescription* pBodyIndexFrameDescription = NULL;
		int nBodyIndexWidth = 0;
		int nBodyIndexHeight = 0;
		UINT nBodyIndexBufferSize = 0;
		BYTE *pBodyIndexBuffer = NULL;

		// get depth frame data

		if (SUCCEEDED(hr))
		{
			hr = pDepthFrame->get_FrameDescription(&pDepthFrameDescription);
		}

		if (SUCCEEDED(hr))
		{
			hr = pDepthFrameDescription->get_Width(&nDepthWidth);
		}

		if (SUCCEEDED(hr))
		{
			hr = pDepthFrameDescription->get_Height(&nDepthHeight);
		}

		if (SUCCEEDED(hr))
		{
			hr = pDepthFrame->AccessUnderlyingBuffer(&nDepthBufferSize, &pDepthBuffer);
		}


		if (SUCCEEDED(hr))
		{

			if (SUCCEEDED(hr)) hr = pInfraredFrame->get_FrameDescription(&pInfraredFrameDescription);
			if (SUCCEEDED(hr)) hr = pInfraredFrameDescription->get_Width(&nInfraredWidth);
			if (SUCCEEDED(hr)) hr = pInfraredFrameDescription->get_Height(&nInfraredHeight);
			if (SUCCEEDED(hr)) hr = pInfraredFrame->AccessUnderlyingBuffer(&nInfraredBufferSize, &pInfraredBuffer);

			if (SUCCEEDED(hr) && pInfraredBuffer && (nInfraredWidth == cInfraredWidth) && (nInfraredHeight == cInfraredHeight))
			{
				RGBQUAD* pDest = m_pInfraredRGBX;

				// end pixel is start + width*height - 1
				const UINT16* pBufferEnd = pInfraredBuffer + (nInfraredWidth * nInfraredHeight);


				int i = 0;
				while ((pInfraredBuffer + i) < pBufferEnd)
				{
					// normalize the incoming infrared data (ushort) to a float ranging from 
					// [InfraredOutputValueMinimum, InfraredOutputValueMaximum] by
					// 1. dividing the incoming value by the source maximum value
					float intensityRatio = static_cast<float>(*(pInfraredBuffer + i)) / InfraredSourceValueMaximum;

					// 2. dividing by the (average scene value * standard deviations)
					intensityRatio /= InfraredSceneValueAverage * InfraredSceneStandardDeviations;

					// 3. limiting the value to InfraredOutputValueMaximum
					intensityRatio = std::min(InfraredOutputValueMaximum, intensityRatio);

					// 4. limiting the lower value InfraredOutputValueMinimym
					intensityRatio = std::max(InfraredOutputValueMinimum, intensityRatio);

					// 5. converting the normalized value to a byte and using the result
					// as the RGB components required by the image
					byte intensity = static_cast<byte>(intensityRatio * 255.0f);
					pDest->rgbRed = intensity;
					pDest->rgbGreen = intensity;
					pDest->rgbBlue = intensity;

					++pDest;
					//++pInfraredBuffer;
					i++;
				}
			}
			SafeRelease(pInfraredFrameDescription);
		}

		// get color frame data

		if (SUCCEEDED(hr))
		{
			hr = pColorFrame->get_FrameDescription(&pColorFrameDescription);
		}

		if (SUCCEEDED(hr))
		{
			hr = pColorFrameDescription->get_Width(&nColorWidth);
		}

		if (SUCCEEDED(hr))
		{
			hr = pColorFrameDescription->get_Height(&nColorHeight);
		}

		if (SUCCEEDED(hr))
		{
			hr = pColorFrame->get_RawColorImageFormat(&imageFormat);
		}

		if (SUCCEEDED(hr))
		{
			if (imageFormat == ColorImageFormat_Bgra)
			{
				hr = pColorFrame->AccessRawUnderlyingBuffer(&nColorBufferSize, reinterpret_cast<BYTE**>(&pColorBuffer));
			}
			else if (m_pColorRGBX)
			{
				pColorBuffer = m_pColorRGBX;
				nColorBufferSize = cColorWidth * cColorHeight * sizeof(RGBQUAD);
				hr = pColorFrame->CopyConvertedFrameDataToArray(nColorBufferSize, reinterpret_cast<BYTE*>(pColorBuffer), ColorImageFormat_Bgra);
			}
			else
			{
				hr = E_FAIL;
			}
		}

		// get body index frame data

		if (SUCCEEDED(hr))
		{
			hr = pBodyIndexFrame->get_FrameDescription(&pBodyIndexFrameDescription);
		}

		if (SUCCEEDED(hr))
		{
			hr = pBodyIndexFrameDescription->get_Width(&nBodyIndexWidth);
		}

		if (SUCCEEDED(hr))
		{
			hr = pBodyIndexFrameDescription->get_Height(&nBodyIndexHeight);
		}

		if (SUCCEEDED(hr))
		{
			hr = pBodyIndexFrame->AccessUnderlyingBuffer(&nBodyIndexBufferSize, &pBodyIndexBuffer);
		}

		// get joint data

		IBody* ppBodies[BODY_COUNT] = { 0 };

		if (SUCCEEDED(hr))
		{
			hr = pBodyFrame->GetAndRefreshBodyData(_countof(ppBodies), ppBodies);
		}

		if (SUCCEEDED(hr))
		{
			ProcessFrame(
				pDepthBuffer, nDepthWidth, nDepthHeight,
				pInfraredBuffer, nInfraredWidth, nInfraredHeight,
				pColorBuffer, nColorWidth, nColorHeight,
				pBodyIndexBuffer, nBodyIndexWidth, nBodyIndexHeight,
				BODY_COUNT, ppBodies);
		}


		for (int i = 0; i < _countof(ppBodies); ++i)
		{
			SafeRelease(ppBodies[i]);
		}
		SafeRelease(pDepthFrameDescription);
		SafeRelease(pColorFrameDescription);
		SafeRelease(pBodyIndexFrameDescription);
	}

	SafeRelease(pDepthFrame);
	SafeRelease(pInfraredFrame);
	SafeRelease(pColorFrame);
	SafeRelease(pBodyIndexFrame);
	SafeRelease(pMultiSourceFrame);
	SafeRelease(pBodyFrame);
}



/// <summary>
/// Initializes the default Kinect sensor
/// </summary>
/// <returns>indicates success or failure</returns>
HRESULT App_predict::InitializeDefaultSensor()
{
	HRESULT hr;

	hr = GetDefaultKinectSensor(&m_pKinectSensor);
	if (FAILED(hr))
	{
		return hr;
	}

	if (m_pKinectSensor)
	{
		// Initialize the Kinect and get coordinate mapper and the frame reader

		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_CoordinateMapper(&m_pCoordinateMapper);
		}

		hr = m_pKinectSensor->Open();

		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->OpenMultiSourceFrameReader(
				FrameSourceTypes::FrameSourceTypes_Depth | FrameSourceTypes::FrameSourceTypes_Color | FrameSourceTypes::FrameSourceTypes_BodyIndex | FrameSourceTypes::FrameSourceTypes_Body | FrameSourceTypes::FrameSourceTypes_Infrared,
				&m_pMultiSourceFrameReader);
		}


		// get the color reader
		IColorFrameSource* pColorFrameSource = NULL;

		hr = m_pKinectSensor->Open();

		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_ColorFrameSource(&pColorFrameSource);
		}

		if (SUCCEEDED(hr))
		{
			hr = pColorFrameSource->OpenReader(&m_pColorFrameReader);
		}

		SafeRelease(pColorFrameSource);


		// get the infrared reader
		IInfraredFrameSource* pInfraredFrameSource = NULL;

		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_InfraredFrameSource(&pInfraredFrameSource);
		}

		if (SUCCEEDED(hr))
		{
			hr = pInfraredFrameSource->OpenReader(&m_pInfraredFrameReader);
		}

		SafeRelease(pInfraredFrameSource);
	}

	return hr;
}


/// <summary>
/// Checks if a specific key has been pressed
/// </summary>
/// <param name="vkcode">keyboard key code</param>
bool App_predict::keyWasPressed(int vkcode)
{
	short ks;
	short LSB;
	short MSB;

	ks = GetAsyncKeyState(vkcode);
	LSB = ks & 1;
	MSB = ks > 0;
	if (LSB || MSB) return true;
	return false;
}


/// <summary>
/// Initialized callibrated coordinate mapper
/// </summary>
void App_predict::InitCallibCoordinateMapper()
{
	// Load display callibration parameters
	cv::FileStorage fs(cDisplayCallibFileName, cv::FileStorage::READ);
	read(fs["H"], m_mColor2DisplayHomography);
	read(fs["MASK"], m_mColorRoiMask);
	fs.release();

	// Load infrared callibration parameters
	cv::FileStorage fs1(cInfraredCallibFileName, cv::FileStorage::READ);
	read(fs1["H"], m_mDefocused2FocusedDepthHomography);
	read(fs1["MASK"], m_mDepthRoiMask);
	fs1.release();
}

int App_predict::two2oneD(int row, int col, int numCols)
{
	return row*numCols + col;
}

int App_predict::one2twoD(int indx, int numCols, char rORc)
{
	if (rORc == 'r') return indx / numCols;
	else return indx%numCols;
}


/// <summary>
/// Get color buffer
/// </summary>
void App_predict::GetColorBuffer()
{
	if (!m_pColorFrameReader)
	{
		return;
	}

	IColorFrame* pColorFrame = NULL;

	HRESULT hr = m_pColorFrameReader->AcquireLatestFrame(&pColorFrame);

	if (SUCCEEDED(hr))
	{
		INT64 nTime = 0;
		IFrameDescription* pFrameDescription = NULL;
		int nWidth = 0;
		int nHeight = 0;
		ColorImageFormat imageFormat = ColorImageFormat_None;
		UINT nBufferSize = 0;
		RGBQUAD *pBuffer = NULL;

		hr = pColorFrame->get_RelativeTime(&nTime);

		if (SUCCEEDED(hr))
		{
			hr = pColorFrame->get_FrameDescription(&pFrameDescription);
		}

		if (SUCCEEDED(hr))
		{
			hr = pFrameDescription->get_Width(&nWidth);
		}

		if (SUCCEEDED(hr))
		{
			hr = pFrameDescription->get_Height(&nHeight);
		}

		if (SUCCEEDED(hr))
		{
			hr = pColorFrame->get_RawColorImageFormat(&imageFormat);
		}

		if (SUCCEEDED(hr))
		{
			if (imageFormat == ColorImageFormat_Bgra)
			{
				hr = pColorFrame->AccessRawUnderlyingBuffer(&nBufferSize, reinterpret_cast<BYTE**>(&pBuffer));
			}
			else if (m_pColorRGBX)
			{
				pBuffer = m_pColorRGBX;
				nBufferSize = cColorWidth * cColorHeight * sizeof(RGBQUAD);
				hr = pColorFrame->CopyConvertedFrameDataToArray(nBufferSize, reinterpret_cast<BYTE*>(pBuffer), ColorImageFormat_Bgra);
			}
			else
			{
				hr = E_FAIL;
			}
		}

		SafeRelease(pFrameDescription);
	}

	SafeRelease(pColorFrame);
}


/// <summary>
/// Get infrared buffer
/// </summary>
void App_predict::GetInfraredBuffer()
{
	if (!m_pInfraredFrameReader)
	{
		return;
	}

	IInfraredFrame* pInfraredFrame = NULL;

	HRESULT hr = m_pInfraredFrameReader->AcquireLatestFrame(&pInfraredFrame);

	if (SUCCEEDED(hr))
	{
		INT64 nTime = 0;
		IFrameDescription* pFrameDescription = NULL;
		int nWidth = 0;
		int nHeight = 0;
		UINT nBufferSize = 0;
		UINT16 *pBuffer = NULL;

		hr = pInfraredFrame->get_RelativeTime(&nTime);

		if (SUCCEEDED(hr)) hr = pInfraredFrame->get_FrameDescription(&pFrameDescription);
		if (SUCCEEDED(hr)) hr = pFrameDescription->get_Width(&nWidth);
		if (SUCCEEDED(hr)) hr = pFrameDescription->get_Height(&nHeight);
		if (SUCCEEDED(hr)) hr = pInfraredFrame->AccessUnderlyingBuffer(&nBufferSize, &pBuffer);

		if (SUCCEEDED(hr) && pBuffer && (nWidth == cInfraredWidth) && (nHeight == cInfraredHeight))
		{
			RGBQUAD* pDest = m_pInfraredRGBX;

			// end pixel is start + width*height - 1
			const UINT16* pBufferEnd = pBuffer + (nWidth * nHeight);

			while (pBuffer < pBufferEnd)
			{
				// normalize the incoming infrared data (ushort) to a float ranging from 
				// [InfraredOutputValueMinimum, InfraredOutputValueMaximum] by
				// 1. dividing the incoming value by the source maximum value
				float intensityRatio = static_cast<float>(*pBuffer) / InfraredSourceValueMaximum;

				// 2. dividing by the (average scene value * standard deviations)
				intensityRatio /= InfraredSceneValueAverage * InfraredSceneStandardDeviations;

				// 3. limiting the value to InfraredOutputValueMaximum
				intensityRatio = std::min(InfraredOutputValueMaximum, intensityRatio);

				// 4. limiting the lower value InfraredOutputValueMinimym
				intensityRatio = std::max(InfraredOutputValueMinimum, intensityRatio);

				// 5. converting the normalized value to a byte and using the result
				// as the RGB components required by the image
				byte intensity = static_cast<byte>(intensityRatio * 255.0f);
				pDest->rgbRed = intensity;
				pDest->rgbGreen = intensity;
				pDest->rgbBlue = intensity;

				++pDest;
				++pBuffer;
			}
		}
		SafeRelease(pFrameDescription);
	}
	SafeRelease(pInfraredFrame);
}

















// show output frames
//cv::imshow(cDisplayWinName, warpedDisplayMaskColor);
//cv::waitKey(1);
//GetColorBuffer();
//cv::Mat tempColorFrame(cv::Size(nColorWidth, nColorHeight), CV_8UC4, reinterpret_cast<void*>(m_pColorRGBX));
//cv::Mat colorFrame;
//cv::Rect cropRect = cv::boundingRect(m_mColorRoiMask);
//cv::Mat cropRef(tempColorFrame, cropRect);
//cropRef.copyTo(colorFrame, cropRef);
//if (mainDisp) cv::imshow(cMainWinName, colorFrame);



/// Detect edges in warped body frame
//int edgeThresh = 1;
//int lowThreshold = 0;
//int const max_lowThreshold = 100;
//int ratio = 3;
//int kernel_size = 3;
//char* window_name = "Edge Map";
//cv::Mat warpedBodyFrameEdges = cv::Mat(nDepthHeight, nDepthWidth, CV_8UC1, cv::Scalar(0));
//cv::Canny(warpedBodyFrame, warpedBodyFrameEdges, lowThreshold, lowThreshold*ratio, kernel_size);





//int bboxTop = faceJointDepthY;
//while (warpedBodyFrame.at<BYTE>(bboxTop, faceJointDepthX) == i && bboxTop > (buffer + 1)) bboxTop--;
//bboxTop -= buffer;
//int bboxBot = faceJointDepthY - bboxTop + faceJointDepthY; 

// get face bbox
/*float faceNeckJointDist = (float)abs((faceJointDepthY - neckJointDepthY) * (faceJointDepthY - neckJointDepthY))*2.0F;
int bboxRight = faceJointDepthX + faceNeckJointDist;
int bboxLeft = faceJointDepthX - faceNeckJointDist;
int bboxTop = faceJointDepthY - faceNeckJointDist;
int bboxBot = faceJointDepthY + faceNeckJointDist;*/

//// get anonymizing face ellipse feat pts
//cv::Mat mAnonymizingFaceRGB = cv::imread("black.PNG");
//float ellipseHeight = mAnonymizingFaceRGB.rows;
//float ellipseWidth = mAnonymizingFaceRGB.cols;
//std::vector<cv::Point2f> anonFaceFeatPoints;
//anonFaceFeatPoints.push_back(cv::Point2f(0,0));
//anonFaceFeatPoints.push_back(cv::Point2f(mAnonymizingFaceRGB.cols - 1, 0));
//anonFaceFeatPoints.push_back(cv::Point2f(0, mAnonymizingFaceRGB.rows - 1));
//anonFaceFeatPoints.push_back(cv::Point2f(mAnonymizingFaceRGB.cols - 1, mAnonymizingFaceRGB.rows - 1));


// Drawings for debug
//DrawPoints(&anonFaceFeatPoints, mAnonymizingFaceRGB, 20, cv::Scalar(255,0,0));
//DrawPoints(&depthFaceFeatPts, warpedDepthFrame, 5, cv::Scalar(UINT16_MAX));
//cv::imshow(cSecondWinName, warpedDepthFrame);
/*cv::Mat temp1 = warpedBodyFrame.clone() * 0;
cv::Mat temp2 = warpedBodyFrame.clone();
cv::Mat mAnonymizingFaceGray = cv::Mat(mAnonymizingFaceRGB.size(), CV_8UC1);
cvtColor(mAnonymizingFaceRGB, mAnonymizingFaceGray, CV_RGB2GRAY);
cv::warpPerspective(mAnonymizingFaceGray, temp1, face2DepthHom, displayMaskDepth.size());
temp2 += temp1;
secDisp = false; cv::imshow(cSecondWinName, warpedDepthFrame * 5);
mainDisp = false; cv::imshow(cMainWinName, temp2);*/

//// get depth rotation
//float faceAngleInDeg = (180.0 / M_PI)*atan(abs(depthFaceFeatPts[0].y - depthFaceFeatPts[3].y) / abs(depthFaceFeatPts[0].x - depthFaceFeatPts[3].x));
//float rotAngleInDeg = (90.0 - faceAngleInDeg);
//if (depthFaceFeatPts[0].x > depthFaceFeatPts[3].x)
//{
//	rotAngleInDeg *= -1;
//}

//// get anonymizing face and its ellipse
//cv::Mat mAnonymizingFaceRGB = cv::imread("black.PNG");
//cv::Mat mRotatedAnonymizingFaceRGB;

//// rotation: rotate anonymizing face
//cv::Point2f src_center(mAnonymizingFaceRGB.cols / 2.0F, mAnonymizingFaceRGB.rows / 2.0F);
//cv::Mat rot_mat = getRotationMatrix2D(src_center, rotAngleInDeg, 1.0);
//cv::Mat mRotatedAnonymizingFace;
//warpAffine(mAnonymizingFaceRGB, mRotatedAnonymizingFaceRGB, rot_mat, mAnonymizingFaceRGB.size());
//cv::Mat mRotatedAnonymizingFaceGray = cv::Mat(mRotatedAnonymizingFaceRGB.size(), CV_8UC1);
//cvtColor(mRotatedAnonymizingFaceRGB, mRotatedAnonymizingFaceGray, CV_RGB2GRAY);
//cv::Mat mRotatedAnonymizingFaceBW = mRotatedAnonymizingFaceGray > 30;

//int anonCenterX = mRotatedAnonymizingFaceBW.cols / 2;
//int anonCenterY = mRotatedAnonymizingFaceBW.rows / 2;
//int buffer = 10;
//
//int anonBboxRight = anonCenterX;
//while (mRotatedAnonymizingFaceBW.at<BYTE>(anonCenterY, anonBboxRight) == i && anonBboxRight < nDepthWidth - (buffer + 1)) anonBboxRight++;
//anonBboxRight += buffer;

//int anonBboxLeft = anonCenterX;
//while (mRotatedAnonymizingFaceBW.at<BYTE>(anonCenterY, anonBboxLeft) == i && anonBboxLeft >(buffer + 1)) anonBboxLeft--;
//anonBboxLeft -= buffer;

//int anonBboxTop = anonCenterY;
//while (mRotatedAnonymizingFaceBW.at<BYTE>(anonBboxTop, anonCenterX) == i && anonBboxTop > (buffer + 1)) anonBboxTop--;
//anonBboxTop -= buffer;

//int anonBboxBot = anonCenterY - anonBboxTop + anonCenterY;


//std::vector<cv::Point2f> anonFacePoints;
//for (int anonFaceY = anonBboxTop; anonFaceY < anonCenterY; anonFaceY++)
//{
//	for (int anonFaceX = anonBboxLeft; anonFaceX < anonBboxRight; anonFaceX++)
//	{
//		if (mRotatedAnonymizingFaceBW.at<byte>(anonFaceY, anonFaceX) == 255)
//		{
//			anonFacePoints.push_back(cv::Point2f(anonFaceX, anonFaceY));

//			int oppPtX = anonCenterX + -1 * (anonCenterX - anonFaceX);
//			int oppPtY = anonCenterY + abs(anonCenterY - anonFaceY);
//			anonFacePoints.push_back(cv::Point2f(oppPtX, oppPtY));
//		}
//	}
//}
//cv::RotatedRect anonFaceEllipse = cv::fitEllipse(anonFacePoints);


//// get anonymizing face and its ellipse
//cv::Mat mAnonymizingFaceRGB = cv::imread("black.PNG");
//cv::Mat mWarpedAnonymizingFaceRGB = mAnonymizingFaceRGB.clone();
//cv::Mat mAnonymizingFaceGray = cv::Mat(mAnonymizingFaceRGB.size(), CV_8UC1);
//cvtColor(mAnonymizingFaceRGB, mAnonymizingFaceGray, CV_RGB2GRAY);
//cv::Mat mAnonymizingFaceBW = mAnonymizingFaceGray > 30;
//std::vector<cv::Point2f> anonFacePoints;
//for (int anonFaceY = 0; anonFaceY < mAnonymizingFaceBW.rows; anonFaceY++)
//{
//	for (int anonFaceX = 0; anonFaceX < mAnonymizingFaceBW.cols; anonFaceX++)
//	{
//		if (mAnonymizingFaceBW.at<byte>(anonFaceY, anonFaceX) == 255)
//		{
//			anonFacePoints.push_back(cv::Point2f(anonFaceX, anonFaceY));
//		}
//	}
//}
//cv::RotatedRect anonFaceEllipse = cv::fitEllipse(anonFacePoints);

/*cv::ellipse(mRotatedAnonymizingFaceRGB, anonFaceEllipse, cv::Scalar(255, 0, 0), 2, 8);
cv::imshow(cMainWinName, warpedDepthFrame*5);
cv::imshow(cSecondWinName, mRotatedAnonymizingFaceRGB);
cv::waitKey(5000);
cv::waitKey(5000);*/

//// get face to depth homography, translation, scale, rotation: get target face bbox in depth
//std::vector<cv::Point2f> anonFaceFeatPts;

//cv::Point2f tempPts[4];
//int sorted[4] = { 0, 1, 2, 3 };
//anonFaceEllipse.points(tempPts);
//anonFaceFeatPts.push_back(tempPts[0]);
//anonFaceFeatPts.push_back(tempPts[1]);
//anonFaceFeatPts.push_back(tempPts[2]);
//anonFaceFeatPts.push_back(tempPts[3]);
//std::stable_sort(anonFaceFeatPts.begin(), anonFaceFeatPts.end(), xComp);
//std::stable_sort(anonFaceFeatPts.begin(), anonFaceFeatPts.end(), yComp);



//cv::Mat face2DepthHom = cv::findHomography(anonFaceFeatPts, depthFaceFeatPts);

//DrawPoints(&anonFaceFeatPts, mAnonymizingFaceRGB, 5, cv::Scalar(255));
//DrawPoints(&depthFaceFeatPts, warpedDepthFrame, 5, cv::Scalar(UINT16_MAX));
/*cv::imshow(cMainWinName, warpedDepthFrame);
cv::imshow(cSecondWinName, mAnonymizingFaceRGB);
cv::waitKey(5000);
cv::waitKey(5000);*/

//cv::Mat temp1 = warpedBodyFrame.clone() * 0;
//cv::Mat temp2 = warpedBodyFrame.clone();
//cv::warpPerspective(mRotatedAnonymizingFaceGray, temp1, face2DepthHom, displayMaskDepth.size());
//temp2 += temp1;
//cv::imshow(cSecondWinName, temp2);

////cv::waitKey(5000);
////cv::waitKey(5000);

//////cv::imshow(cSecondWinName, test);

//////int bboxRight = faceJointDepthX;
//////int iterY = faceJointDepthY;
//////while (test.at<BYTE>(iterY, bboxRight) == 255  && bboxRight < nDepthWidth - 2)
//////{
//////	bboxRight += 1;
//////	iterY += abs(faceJointDepthY - neckJointDepthY);
//////}

//////std::cout << bboxRight << " - " << iterY;
//////
//////cv::circle(warpedDepthFrame, cvPoint(bboxRight, iterY), 10, cvScalar(UINT16_MAX), -1, 8, 0);

//////cv::imshow(cMainWinName, warpedDepthFrame);
//////cv::waitKey(5000);

//////int bboxLeft = faceJointDepthX;
//////while (warpedBodyFrame.at<BYTE>(faceJointDepthY, bboxLeft) == i && bboxLeft < nDepthWidth - 2) bboxLeft--;

//////int bboxTop = faceJointDepthY;
//////while (warpedBodyFrame.at<BYTE>(bboxTop, faceJointDepthX) == i && bboxTop > 1) bboxTop--;

//////int bboxBot = faceJointDepthY - bboxTop + faceJointDepthY;


////// draw rect onto warped depth frame
////cv::Rect faceBbox = cv::Rect(bboxLeft, bboxTop, bboxRight - bboxLeft + 1, bboxBot - bboxTop + 1);
////cv::rectangle(warpedDepthFrame, faceBbox, cv::Scalar(UINT16_MAX));

////// get rotation
////float faceAngleInDeg = (180.0 / M_PI)*atan(abs(pBodyJoints[i][JointType_Neck].y - pBodyJoints[i][JointType_Head].y) / abs(pBodyJoints[i][JointType_Head].x - pBodyJoints[i][JointType_Neck].x));
////float rotAngleInDeg = (90.0 - faceAngleInDeg);
////if (pBodyJoints[i][JointType_Head].x > pBodyJoints[i][JointType_Neck].x)
////{
////	rotAngleInDeg *= -1;
////}

////// rotation: rotate anonymizing face
////cv::Point2f src_center(mAnonymizingFace.cols / 2.0F, mAnonymizingFace.rows / 2.0F);
////cv::Mat rot_mat = getRotationMatrix2D(src_center, rotAngleInDeg, 1.0);
////cv::Mat mRotatedAnonymizingFace;
////warpAffine(mAnonymizingFace, mRotatedAnonymizingFace, rot_mat, mAnonymizingFace.size());

////// scaling: resize anonymizing face bbox to target face bbox
////cv::Mat mRotAndResAnonymizingFace = cv::Mat(faceBbox.height, faceBbox.width, CV_8UC3, cv::Scalar(0, 0, 0));
////cv::resize(mRotatedAnonymizingFace, mRotAndResAnonymizingFace, mRotAndResAnonymizingFace.size());

////// translation: embedd anonymizing face into display mask "depth frame"
////cv::Mat displayMaskDepthSingleFace = cv::Mat(nDepthHeight, nDepthWidth, CV_8UC3, cv::Scalar(0, 0, 0));
////mRotAndResAnonymizingFace.copyTo(displayMaskDepthSingleFace(faceBbox));
////mRotAndResAnonymizingFace.copyTo(displayMaskDepth(faceBbox));

////// get face to depth homography, translation, scale, rotation: get target face bbox in depth
////std::vector<cv::Point2f> faceRotBboxCorners = { cv::Point(0, 0), cv::Point(0, mRotatedAnonymizingFace.cols - 1), cv::Point(mRotatedAnonymizingFace.rows - 1, 0), cv::Point(mRotatedAnonymizingFace.rows - 1, mRotatedAnonymizingFace.cols - 1) };
////std::vector<cv::Point2f> faceDepthBboxCorners = { cv::Point(bboxLeft, bboxTop), cv::Point(bboxLeft, bboxBot), cv::Point(bboxRight, bboxTop), cv::Point(bboxRight, bboxBot) };
////cv::Mat face2DepthHom = cv::findHomography(faceRotBboxCorners, faceDepthBboxCorners);

////cv::warpPerspective(mRotatedAnonymizingFace, displayMaskDepth, face2DepthHom, displayMaskDepth.size());

////// get target face bbox in depth
//////cv::Mat croppedRef(pFaceMasks[i], faceBbox);
//////croppedRef.copyTo(pFaces[i], croppedRef);

////// get scale
//////float yScale = mRotatedAnonymizingFace.rows / faceBbox.height;
//////float xScale = mRotatedAnonymizingFace.cols / faceBbox.width;

////////// get translation
//////int xTrans = (faceBbox.x + faceBbox.width / 2.0F) - (mAnonymizingFace.rows / 2.0F);
//////int yTrans = (faceBbox.y + faceBbox.height / 2.0F) - (mAnonymizingFace.cols / 2.0F);


//////cv::Mat face2DepthHom = cv::Mat(3, 3, CV_32F);
//////face2DepthHom.at<float>(0, 0) = xScale * cos(rotAngleInDeg * M_PI / 180.0);
//////face2DepthHom.at<float>(0, 1) = xScale * -1 * sin(rotAngleInDeg * M_PI / 180.0);
//////face2DepthHom.at<float>(0, 2) = xTrans;
//////face2DepthHom.at<float>(1, 0) = yScale * sin(rotAngleInDeg * M_PI / 180.0);
//////face2DepthHom.at<float>(1, 1) = yScale * cos(rotAngleInDeg * M_PI / 180.0);
//////face2DepthHom.at<float>(1, 2) = yTrans;
//////face2DepthHom.at<float>(2, 0) = 0.0;
//////face2DepthHom.at<float>(2, 1) = 0.0;
//////face2DepthHom.at<float>(2, 2) = 1.0;

//////cv::Mat mWarpedAnonymizingFace = cv::Mat(nDepthHeight, nDepthWidth, CV_8UC3, cv::Scalar(0, 0, 0));
//////cv::warpPerspective(mAnonymizingFace, mWarpedAnonymizingFace, face2DepthHom, mWarpedAnonymizingFace.size());

////// scaling: resize anonymizing face bbox to target face bbox
//////cv::resize(mRotatedAnonymizingFace, mfinalAnonymizingFace, mfinalAnonymizingFace.size());

////// translation: embedd anonymizing face into display mask "depth frame"
//////cv::Mat displayMaskDepthSingleFace = cv::Mat(nDepthHeight, nDepthWidth, CV_8UC3, cv::Scalar(0, 0, 0));
//////mRotAndResAnonymizingFace.copyTo(displayMaskDepthSingleFace(faceBbox));
//////mRotAndResAnonymizingFace.copyTo(displayMaskDepth(faceBbox));



// map display mask "depth frame" to display mask "color frame"
//std::vector<cv::Point2f> homDepthPoints;
//std::vector<cv::Point2f> homColorPoints;
//cv::Rect faceBbox = cv::Rect(bboxLeft, bboxTop, bboxRight - bboxLeft + 1, bboxBot - bboxTop + 1);

//for (int depthY = faceBbox.y; depthY < faceBbox.y + faceBbox.height - 1; depthY++)
//{
//	for (int depthX = faceBbox.x; depthX < faceBbox.x + faceBbox.width - 1; depthX++)
//	{
//		cv::Point2f depthPt = cv::Point2f(depthX, depthY);

//		if (warpedBodyFrame.at<byte>(depthY, depthX) == i)
//		{
//			ColorSpacePoint p = m_pColorCoordinates[two2oneD(depthY, depthX, nDepthWidth)];
//			int colorX = static_cast<int>(p.X + 0.5f);
//			int colorY = static_cast<int>(p.Y + 0.5f);

//			// if valid depth to color mapping
//			if (p.X != -std::numeric_limits<float>::infinity() && p.Y != -std::numeric_limits<float>::infinity() &&
//				(colorX >= 0 && colorX < nColorWidth) && (colorY >= 0 && colorY < nColorHeight))
//			{
//				homDepthPoints.push_back(depthPt);
//				homColorPoints.push_back(cv::Point2f(colorX, colorY));
//			}
//		}
//	}
//}

//cv::Mat depth2ColorHom = cv::findHomography(homDepthPoints, homColorPoints);
//cv::Mat temp = cv::Mat(nColorHeight, nColorWidth, CV_8UC3, cv::Scalar(0, 0, 0));
//cv::warpPerspective(mAnonymizingFaceRGB, temp, depth2ColorHom, temp.size());
//displayMaskColor += temp;
//cv::imshow(cSecondWinName, displayMaskColor);

//////cv::Mat depth2ColorHom = cv::findHomography(homDepthPoints, homColorPoints);
//////cv::Mat temp = cv::Mat(nColorHeight, nColorWidth, CV_8UC3, cv::Scalar(0, 0, 0));
//////cv::warpPerspective(displayMaskDepthSingleFace, temp, depth2ColorHom, temp.size());
//////displayMaskColor += temp;
//////cv::imshow(cSecondWinName, displayMaskColor);




//if (SUCCEEDED(hr))
//{
//	cv::Vec4b zeroVec = cv::Vec4b((byte)0);

//	// loop over output pixels
//	for (int colorIndex = 0; colorIndex < (nColorWidth * nColorHeight); colorIndex++)
//	{

//		DepthSpacePoint p = m_pDepthCoordinates[colorIndex];
//		int depthX = static_cast<int>(p.X + 0.5f);
//		int depthY = static_cast<int>(p.Y + 0.5f);

//		// if valid depth to color mapping
//		if ((depthX >= 0 && depthX < nDepthWidth) && (depthY >= 0 && depthY < nDepthHeight) && p.X != -std::numeric_limits<float>::infinity() && p.Y != -std::numeric_limits<float>::infinity())
//		{
//			// player index (0-5)
//			BYTE player = pBodyIndexBuffer[depthX + (depthY * nDepthWidth)];

//			if (player >= 0 && player <= 5)
//			{
//				if (!pBodyJoints[player].empty())
//				{
//					int faceJointDepthX = static_cast<int>(pBodyJoints[player][JointType_Head].x + 0.5f);
//					int faceJointDepthY = static_cast<int>(pBodyJoints[player][JointType_Head].y + 0.5f);
//					UINT16 faceJointDepth = pDepthBuffer[faceJointDepthX + (faceJointDepthY * nDepthWidth)];
//					UINT16 depth = pDepthBuffer[depthX + (depthY * nDepthWidth)];
//					int depthThreshold = 100;

//					// if point above neck and within depth threshold of center face pixel
//					if (depthY < pBodyJoints[player][JointType_Neck].y && abs(faceJointDepth - depth) <= depthThreshold)
//					{
//						
//						int colorY = one2twoD(colorIndex, nColorWidth, 'r');
//						int colorX = one2twoD(colorIndex, nColorWidth, 'c');

//						//displayMask.at<cv::Vec4b>(colorY, colorX) = zeroVec;
//						//pFaceMasks[player].at<byte>(depthY, depthX) = (byte)255;
//						
//						if (depthX < pFaceBBoxPts[player][0].x)
//							pFaceBBoxPts[player][0].x = depthX;
//						else if (depthX > pFaceBBoxPts[player][1].x)
//							pFaceBBoxPts[player][1].x = depthX;

//						if (depthY < pFaceBBoxPts[player][0].y)
//							pFaceBBoxPts[player][0].y = depthY;
//						else if (depthY > pFaceBBoxPts[player][1].y)
//							pFaceBBoxPts[player][1].y = depthY;
//					}
//				}
//			}
//		}
//	}
//}


//if (SUCCEEDED(hr))
//{
//	for (int depthY = 0; depthY < nDepthHeight; depthY++)
//	{
//		for (int depthX = 0; depthX < nDepthWidth; depthX++)
//		{
//			// player index (0-5)
//			BYTE player = pBodyIndexBuffer[depthX + (depthY * nDepthWidth)];

//			if (player >= 0 && player <= 5)
//			{
//				if (!pBodyJoints[player].empty())
//				{
//					int faceJointDepthX = static_cast<int>(pBodyJoints[player][JointType_Head].x + 0.5f);
//					int faceJointDepthY = static_cast<int>(pBodyJoints[player][JointType_Head].y + 0.5f);
//					UINT16 faceJointDepth = pDepthBuffer[faceJointDepthX + (faceJointDepthY * nDepthWidth)];
//					int faceDepthThreshold = 100;

//					int neckJointDepthX = static_cast<int>(pBodyJoints[player][JointType_Neck].x + 0.5f);
//					int neckJointDepthY = static_cast<int>(pBodyJoints[player][JointType_Neck].y + 0.5f);
//					UINT16 neckJointDepth = pDepthBuffer[neckJointDepthX + (neckJointDepthY * nDepthWidth)];
//					int neckDepthThreshold = neckJointDepth - faceJointDepth;

//					UINT16 depth = pDepthBuffer[depthX + (depthY * nDepthWidth)];
//					

//					// if point above neck and within depth threshold of center face pixel
//					if (depthY < neckJointDepthY)// && abs(depthX - faceJointDepthX) < (1/depth)*100)//abs(faceJointDepth - depth) <= depthThreshold)
//					{
//						if (depthX < pFaceBBoxPts[player][0].x)
//							pFaceBBoxPts[player][0].x = depthX;
//						else if (depthX > pFaceBBoxPts[player][1].x)
//							pFaceBBoxPts[player][1].x = depthX;

//						if (depthY < pFaceBBoxPts[player][0].y)
//							pFaceBBoxPts[player][0].y = depthY;
//						else if (depthY > pFaceBBoxPts[player][1].y)
//							pFaceBBoxPts[player][1].y = depthY;
//					}
//				}
//			}
//		}
//	}
//}



//// map display mask "depth frame" to display mask "color frame"
//if (SUCCEEDED(hr))
//{
//	cv::Vec3b pixelVal;

//	// loop over display pixels
//	for (int colorIndex = 0; colorIndex < (nColorWidth * nColorHeight); colorIndex++)
//	{
//		
//		int colorY = one2twoD(colorIndex, nColorWidth, 'r');
//		int colorX = one2twoD(colorIndex, nColorWidth, 'c');

//		DepthSpacePoint p = m_pDepthCoordinates[colorIndex];
//		int depthX = static_cast<int>(p.X + 0.5f);
//		int depthY = static_cast<int>(p.Y + 0.5f);

//		if (depthX >= 0 && depthX < nDepthWidth && depthY >= 0 && depthY < nDepthHeight)
//		{
//			pixelVal = displayMaskDepth.at<cv::Vec3b>(depthY, depthX);
//			int colorY = one2twoD(colorIndex, nColorWidth, 'r');
//			int colorX = one2twoD(colorIndex, nColorWidth, 'c');
//			displayMaskColor.at<cv::Vec3b>(colorY, colorX) = pixelVal;
//		}
//	}
//}
