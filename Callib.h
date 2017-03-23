//------------------------------------------------------------------------------
// Callib.h
// 
// University of Florida
// Francesco Pittaluga
// f.pittaluga@ufl.edu
//------------------------------------------------------------------------------

#pragma once

#include "stdafx.h"
#include <Windows.h>
#include <Kinect.h>
#include <opencv2\opencv.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>


// InfraredSourceValueMaximum is the highest value that can be returned in the InfraredFrame.
// It is cast to a float for readability in the visualization code.
#define InfraredSourceValueMaximum static_cast<float>(USHRT_MAX)
//#define InfraredSourceValueMaximum static_cast<float>(MAXUINT16)


// The InfraredOutputValueMinimum value is used to set the lower limit, post processing, of the
// infrared data that we will render.
// Increasing or decreasing this value sets a brightness "wall" either closer or further away.
#define InfraredOutputValueMinimum 0.05f 

// The InfraredOutputValueMaximum value is the upper limit, post processing, of the
// infrared data that we will render.
#define InfraredOutputValueMaximum 1.0f

// The InfraredSceneValueAverage value specifies the average infrared value of the scene.
// This value was selected by analyzing the average pixel intensity for a given scene.
// Depending on the visualization requirements for a given application, this value can be
// hard coded, as was done here, or calculated by averaging the intensity for each pixel prior
// to rendering.
#define InfraredSceneValueAverage 0.08f

/// The InfraredSceneStandardDeviations value specifies the number of standard deviations
/// to apply to InfraredSceneValueAverage. This value was selected by analyzing data
/// from a given scene.
/// Depending on the visualization requirements for a given application, this value can be
/// hard coded, as was done here, or calculated at runtime.
#define InfraredSceneStandardDeviations 3.0f

class Callib
{

	static const int        cInfraredWidth = 512;
	static const int        cInfraredHeight = 424;
	static const int        cColorWidth = 1920;
	static const int        cColorHeight = 1080;
	static const int        cDepthWidth = 512;
	static const int        cDepthHeight = 424;
	static const int		cVK_C = 0x43;
	static const int        cDisplayMonitor = 0;
	static const int		cOutputMonitor = 1;

public:
	/// <summary>
	/// Constructor
	/// </summary>
	Callib(std::string mainWinName, std::string sideWinName, std::string dispWinName, std::string dispCallibFileName, std::string infraredCallibFileName);

	/// <summary>
	/// Destructor
	/// </summary>
	~Callib();

	/// <summary>
	/// Run application
	/// </summary>
	void Run();

private:

	// Constants
	std::string				cDisplayCallibFileName;
	std::string				cInfraredCallibFileName;
	std::string				cMainWinName;
	std::string				cDisplayWinName;
	std::string				cSecondWinName;

	// Kinect 
	IKinectSensor*			m_pKinectSensor;
	BOOLEAN					m_WasKinectOpenWhenInitialized;

	// Kinect readers
	IMultiSourceFrameReader*m_pMultiSourceFrameReader;
	IInfraredFrameReader*	m_pInfraredFrameReader;
	IColorFrameReader*		m_pColorFrameReader;
	ICoordinateMapper*      m_pCoordinateMapper;

	// Direct2D
	RGBQUAD*				m_pInfraredRGBX;
	RGBQUAD*		        m_pColorRGBX;

	// Callibrated Coordinate Mapper
	cv::Mat					m_Color2DisplayHomography;
	cv::Mat					m_Defocused2FocusedInfraredHomography;
	std::vector<cv::Point2f>m_ColorROI;

	/// <summary>
	/// Initializes the default Kinect sensors
	/// </summary>
	/// <returns>indicates success or failure</returns>
	HRESULT					InitializeDefaultKinectSensors();

	/// <summary>
	/// Initializes the console window
	/// </summary>
	/// <returns>indicates success or failure</returns>
	void					InitConsoleWindow();

	/// <summary>
	/// Create new display window
	/// </summary>
	/// <returns>indicates success or failure</returns>
	void					cvNamedDisplayWindow(const char * dispWinName);

	/// <summary>
	/// Creates new main output window
	/// </summary>
	/// <returns>indicates success or failure</returns>
	void					cvNamedMainWindow(const char * mainWinName);

	/// <summary>
	/// Get infrared buffer
	/// </summary>
	void					GetInfraredBuffer();

	/// <summary>
	/// Get infrared buffer as Mat
	/// </summary>
	cv::Mat					GetInfraredBufferAsMat();

	/// <summary>
	/// Get color buffer
	/// </summary>
	void					GetColorBuffer();

	/// <summary>
	/// Returns color buffer frame as a mat
	/// </summary>
	cv::Mat					GetColorBufferAsMat();

	/// <summary>
	/// Run infrared lensless/lenslet callibration
	/// </summary>
	void					RunInfraredCallibration();

	/// <summary>
	/// Display/Beamsplitter/Color Camera Callibration
	/// </summary>
	void					RunDisplayCallibration();

	/// <summary>
	/// Auxilliary Functions
	/// </summary>
	bool					keyWasPressed(int vkcode);
	void					DrawPoints(std::vector<cv::Point2f>* points, cv::Mat frame, int radius = 5, cv::Scalar color = CV_RGB(0, 0, 100));
	template <typename T, typename M> void OneDByteArray2TwoDMat(T oneD, M twoD);
	void					rgbquad2RGBMat(RGBQUAD* r, cv::Mat m);
	void					SetWindow2Rect(HWND winHandle, RECT rect);
	void					DrawQuadrilateral(cv::Mat frame, cv::Point2f topLeft, cv::Point2f bottomLeft, cv::Point2f topRight, cv::Point2f BotttomRight, cv::Scalar color, int thickness);
	static void				onMouse(int evt, int x, int y, int flags, void* param);
	static BOOL CALLBACK	MonitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData);
	int						two2oneD(int row, int col, int numCols);
	int						one2twoD(int indx, int numCols, char rORc);
};

