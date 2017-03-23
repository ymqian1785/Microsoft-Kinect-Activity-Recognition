//------------------------------------------------------------------------------
// AnonymousKinect_Application.h
// 
// University of Florida
// Francesco Pittaluga
// f.pittaluga@ufl.edu
//------------------------------------------------------------------------------

#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include <shlobj.h>
#include <wchar.h>
#include <devicetopology.h>
#include <Functiondiscoverykeys_devpkey.h>
#include <Kinect.h>
#include "ImageRenderer.h"
#include <opencv2\opencv.hpp>
#include <vector>
#include <iostream>
#include "SVMachine.h"

#ifndef HINST_THISCOMPONENT
EXTERN_C IMAGE_DOS_HEADER __ImageBase;
#define HINST_THISCOMPONENT ((HINSTANCE)&__ImageBase)
#endif

# define M_PI           3.14159265358979323846  /* pi */
#define FRAME_RANGE 25 // Number of frames to be captured
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

// InfraredSourceValueMaximum is the highest value that can be returned in the InfraredFrame.
// It is cast to a float for readability in the visualization code.
#define InfraredSourceValueMaximum static_cast<float>(USHRT_MAX)

// The InfraredOutputValueMinimum value is used to set the lower limit, post processing, of the
// infrared data that we will render.
// Increasing or decreasing this value sets a brightness "wall" either closer or further away.
#define InfraredOutputValueMinimum 0.01f 

// The InfraredOutputValueMaximum value is the upper limit, post processing, of the
// infrared data that we will render.
#define InfraredOutputValueMaximum 1.0f

// The InfraredSceneValueAverage value specifies the average infrared value of the scene.
// This value was selected by analyzing the average pixel intensity for a given scene.
// Depending on the visualization requirements for a given application, this value can be
// hard coded, as was done here, or calculated by averaging the intensity for each pixel prior
// to rendering.
#define InfraredSceneValueAverage 0.03f

/// The InfraredSceneStandardDeviations value specifies the number of standard deviations
/// to apply to InfraredSceneValueAverage. This value was selected by analyzing data
/// from a given scene.
/// Depending on the visualization requirements for a given application, this value can be
/// hard coded, as was done here, or calculated at runtime.
#define InfraredSceneStandardDeviations 3.0f

////
////  A wave file consists of:
////
////  RIFF header:    8 bytes consisting of the signature "RIFF" followed by a 4 byte file length.
////  WAVE header:    4 bytes consisting of the signature "WAVE".
////  fmt header:     4 bytes consisting of the signature "fmt " followed by a WAVEFORMATEX 
////  WAVEFORMAT:     <n> bytes containing a waveformat structure.
////  DATA header:    8 bytes consisting of the signature "data" followed by a 4 byte file length.
////  wave data:      <m> bytes containing wave data.
////
//
////  Header for a WAV file - we define a structure describing the first few fields in the header for convenience.
//struct WAVEHEADER
//{
//	DWORD   dwRiff;                     // "RIFF"
//	DWORD   dwSize;                     // Size
//	DWORD   dwWave;                     // "WAVE"
//	DWORD   dwFmt;                      // "fmt "
//	DWORD   dwFmtSize;                  // Wave Format Size
//};
//
////  Static RIFF header, we'll append the format to it.
//const BYTE WaveHeaderTemplate[] =
//{
//	'R', 'I', 'F', 'F', 0x00, 0x00, 0x00, 0x00, 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ', 0x00, 0x00, 0x00, 0x00
//};
//
////  Static wave DATA tag.
//const BYTE WaveData[] = { 'd', 'a', 't', 'a' };

class App:public SVMachine
{
	static const int        cDepthWidth = 512;
	static const int        cDepthHeight = 424;
	static const int        cColorWidth = 1920;
	static const int        cColorHeight = 1080;
	static const int        cInfraredWidth = 512;
	static const int        cInfraredHeight = 424;

	

public:
	/// <summary>
	/// Constructor
	/// </summary>
	App(std::string mainWinName, std::string secondWinName, std::string dispWinName, std::string dispCallibFileName, std::string infraredCallibFileName);

	/// <summary>
	/// Destructor
	/// </summary>
	~App();

	/// <summary>
	/// Handles window messages, passes most to the class instance to handle
	/// </summary>
	/// <param name="hWnd">window message is for</param>
	/// <param name="uMsg">message</param>
	/// <param name="wParam">message data</param>
	/// <param name="lParam">additional message data</param>
	/// <returns>result of message processing</returns>
	static LRESULT CALLBACK MessageRouter(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

	/// <summary>
	/// Handle windows messages for a class instance
	/// </summary>
	/// <param name="hWnd">window message is for</param>
	/// <param name="uMsg">message</param>
	/// <param name="wParam">message data</param>
	/// <param name="lParam">additional message data</param>
	/// <returns>result of message processing</returns>
	LRESULT CALLBACK        DlgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

	/// <summary>
	/// Creates the main window and begins processing
	/// </summary>
	/// <param name="hInstance"></param>
	/// <param name="nCmdShow"></param>
	std::vector<float>                Run();

private:

	// Constants
	std::string				cDisplayCallibFileName;
	std::string				cInfraredCallibFileName;
	std::string				cMainWinName;
	std::string				cDisplayWinName;
	std::string				cSecondWinName;
	const int TargetLatency = 20;                        // Number of milliseconds of acceptable lag between live sound being produced and recording operation.

	// Kinect Objects
	IKinectSensor*          m_pKinectSensor;
	ICoordinateMapper*      m_pCoordinateMapper;
	IMultiSourceFrameReader*m_pMultiSourceFrameReader;
	IColorFrameReader*		m_pColorFrameReader;
	IInfraredFrameReader*	m_pInfraredFrameReader;
	/*IMMDevice *device = NULL;*/
	HANDLE waveFile = INVALID_HANDLE_VALUE;
	/*CWASAPICapture *capturer = NULL;*/

	// Sensor Data
	DepthSpacePoint*        m_pDepthCoordinates;
	ColorSpacePoint*        m_pColorCoordinates;
	RGBQUAD*                m_pColorRGBX;
	RGBQUAD*				m_pInfraredRGBX;

	// Callibration
	cv::Mat					m_mColorRoiMask;
	cv::Mat					m_mDepthRoiMask;
	cv::Mat					m_mColor2DisplayHomography;
	cv::Mat					m_mDefocused2FocusedDepthHomography;

	// Face DB
	std::vector<cv::Point2f>m_vAnonFaceFeatPts;

	// Data log
	char expDir[50];
	char depthDir[50];
	char infraredDir[50];
	char bodyDir[50];
	char colorDir[50];
	char bodyIndexDir[50];
	std::ofstream myfile;
	bool					m_bSaveOutputFrames;
	std::vector<cv::Mat>	m_vOutputFrames;
	std::vector<cv::Mat>	m_vInfraredFrames;

	// Frame Processing Paramaters
	int frameNum;
	std::vector<float> skelVec;
	std::vector<float> depthVec;
	std::vector<float> featVec;
	cv::Mat trainData;
	cv::Mat labels;
	bool flag;
	
	/// <summary>
	/// Main processing function
	/// </summary>
	void                    Update();
	void                    Init_Save(int activityNum);
	void                    Save(int activityNum);
	/// <summary>
	/// Initializes the default Kinect sensor
	/// </summary>
	/// <returns>S_OK on success, otherwise failure code</returns>
	HRESULT                 InitializeDefaultSensor();
	/*HRESULT                 GetKinectAudioDevice(IMMDevice **ppDevice);*/
	/*HRESULT                 WriteWaveHeader(HANDLE waveFile, const WAVEFORMATEX *pWaveFormat, DWORD dataSize);*/
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
	void                ProcessFrame(
		UINT16* pDepthBuffer, int nDepthWidth, int nDepthHeight,
		UINT16* pInfraredBuffer, int nInfraredWidth, int nInfraredHeight,
		RGBQUAD* pColorBuffer, int nColorWidth, int nColorHeight,
		BYTE* pBodyIndexBuffer, int nBodyIndexWidth, int nBodyIndexHeight,
		int nBodyCount, IBody** ppBodies);

	/// <summary>
	/// Get color buffer
	/// </summary>
	void					GetColorBuffer();

	/// <summary>
	/// Get infrared buffer
	/// </summary>
	void					GetInfraredBuffer();

	/// <summary>
	/// Initialized callibrated coordinate mapper
	/// </summary>
	void					InitCallibCoordinateMapper();

	/// <summary>
	/// Initializes the face DB 
	/// </summary>
	void					InitFaceDB();


	void					LogExecutionData();

	/// <summary>
	/// 
	/// </summary>
	static BOOL CALLBACK	MonitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData);
	bool					keyWasPressed(int vkcode);
	int						two2oneD(int row, int col, int numCols);
	int						one2twoD(int indx, int numCols, char rORc);
};