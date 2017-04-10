//------------------------------------------------------------------------------
// AnonymousKinect.cpp
// 
// University of Florida
// Francesco Pittaluga
// f.pittaluga@ufl.edu
// Aashik Nagadikeri Harish
// aashikgowda@ufl.edu
//------------------------------------------------------------------------------

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <conio.h>
#include <stdio.h>
#include <time.h> 

#include "AnonymousKinect.h"
#include "Callib.h"
#include "App.h"
// #include "App_predict.h"
#include "SVMachine.h"

using namespace std;

#define KEYDOWN(vkcode) (GetAsyncKeyState(vkcode) & 0x8000 ? true : false)
#define VK_1 0x31
#define VK_2 0x32
#define VK_3 0x33
#define VK_4 0x34
#define DISPLAY_CALLIB_FILE_NAME "DisplayCallib.yml"
#define DEPTH_CALLIB_FILE_NAME   "InfraredCallib.yml"
#define CONSOLE_NAME	         "Anonymous Kinect"
#define MAIN_WINDOW_NAME         "System Output"
#define DISPLAY_WINDOW_NAME      "Display"
#define SECOND_WINDOW_NAME       "Raw Infrared"

/// <swummary>
/// Constructor
/// </summary>
AnonymousKinect::AnonymousKinect() :
cDisplayCallibFileName(DISPLAY_CALLIB_FILE_NAME),
cInfraredCallibFileName(DEPTH_CALLIB_FILE_NAME),
cMainWindowName(MAIN_WINDOW_NAME),
cDisplayWindowName(DISPLAY_WINDOW_NAME),
cSecondWindowName(SECOND_WINDOW_NAME)
{
}

/// <summary>
/// Destructor
/// </summary>
AnonymousKinect::~AnonymousKinect()
{
}

/// <summary>
/// Entry point for application
/// </summary>
int main()
{
	AnonymousKinect AK;
	AK.Run();

	return 0;
}

/// <summary>
/// Runs application
/// </summary>
/// <returns>indicates success or failure</returns>
void AnonymousKinect::Run()
{
	std::string m_MainWinHwnd = "Output";
	std::vector<float> train_data;
	InitConsoleWindow();
	InitSecondWindow();
	InitMainWindow();

	App    Application (cMainWindowName, cSecondWindowName, cDisplayWindowName, cDisplayCallibFileName, cInfraredCallibFileName);
	// App_predict   Prediction(cMainWindowName, cSecondWindowName, cDisplayWindowName, cDisplayCallibFileName, cInfraredCallibFileName);
	SVMachine    Support_vector;
	char menuText[] = "----------------------------------------------------------------------------\n"
		"*** Main Menu (Press # to select option) *** \n"
	 	"----------------------------------------------------------------------------\n"
		"(1) Quit.\n"
		"(2) Recording Data Menu.\n"
		"(3) SVM Menu.\n"
		"(4) Activity Prediction Menu.\n";

	cout << menuText;

	while (true)
	{
		if (KEYDOWN(VK_ESCAPE) || KEYDOWN(VK_1)) {
			return;
		}
		else if (KEYDOWN(VK_2)) {
			Sleep(100);
			system("cls");
			train_data = Application.Run();
			//cout << train_data.size();
			system("cls");
			std::cout << menuText;
			Sleep(100);
		}
		else if (KEYDOWN(VK_3)) {
			Sleep(100);
			system("cls");
			Support_vector.Run(train_data);
			system("cls");
			std::cout << menuText;
			Sleep(100);	
		}
		else if (KEYDOWN(VK_4)) {
			Sleep(100);
			system("cls");
		//	Prediction.Run();
			system("cls");
			std::cout << menuText;
			Sleep(100);
		}
	}
}


/// <summary>
/// Initializes the console window
/// </summary>
/// <returns>indicates success or failure</returns>
void AnonymousKinect::InitConsoleWindow()
{
	// Get monitor info
	vector<MONITORINFO> *temp_vMonitorInfo = new vector<MONITORINFO>;
	EnumDisplayMonitors(NULL, NULL, MonitorEnumProc, (LPARAM)temp_vMonitorInfo);
	RECT outputMonitorRect = temp_vMonitorInfo->at(cOutputMonitor).rcWork;
	int outputMonitorWidth = outputMonitorRect.right - outputMonitorRect.left;
	int outputMonitorHeight = outputMonitorRect.bottom - outputMonitorRect.top;

	// Set console size/pos
	int consoleX1 = outputMonitorRect.left + outputMonitorWidth / 3 * 2;
	int consoleY1 = outputMonitorRect.top + outputMonitorHeight / 2;
	int consoleX2 = outputMonitorRect.left + outputMonitorWidth;
	int consoleY2 = outputMonitorRect.top + outputMonitorHeight + 10;
	RECT consoleWinRect = { consoleX1, consoleY1, consoleX2, consoleY2 };
	SetWindow2Rect(GetConsoleWindow(), consoleWinRect);

	// Set console title
	SetConsoleTitleA(CONSOLE_NAME);

	delete(temp_vMonitorInfo);
}


/// <summary>
/// Initializes the console window
/// </summary>
/// <returns>indicates success or failure</returns>
void AnonymousKinect::InitDisplayWindow()
{
	// Get monitor info
	vector<MONITORINFO> *temp_vMonitorInfo = new vector<MONITORINFO>;
	EnumDisplayMonitors(NULL, NULL, MonitorEnumProc, (LPARAM)temp_vMonitorInfo);
	RECT displayWinRect = temp_vMonitorInfo->at(cDisplayMonitor).rcWork;
	displayWinRect.left = displayWinRect.left - 1;
	displayWinRect.top = displayWinRect.top - 1;

	// Init display window
	cvNamedWindow(cDisplayWindowName.c_str(), CV_WINDOW_NORMAL);
	HWND dispWinHwnd = FindWindowA(0, cDisplayWindowName.c_str());
	SetWindowLongPtr(dispWinHwnd, GWL_STYLE, WS_SYSMENU | WS_POPUP | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE);
	SetWindow2Rect(dispWinHwnd, displayWinRect);
	cv::imshow(cDisplayWindowName.c_str(), cv::Mat(1, 1, CV_8UC3, cv::Scalar(0, 0, 0)));
	cv::waitKey(33);

	delete(temp_vMonitorInfo);
}

/// <summary>
/// Initializes the main window
/// </summary>
/// <returns>indicates success or failure</returns>
void AnonymousKinect::InitMainWindow()
{
	// Get monitor info
	vector<MONITORINFO> *temp_vMonitorInfo = new vector<MONITORINFO>;
	EnumDisplayMonitors(NULL, NULL, MonitorEnumProc, (LPARAM)temp_vMonitorInfo);
	RECT outputMonitorRect = temp_vMonitorInfo->at(cOutputMonitor).rcWork;
	int outputMonitorWidth = outputMonitorRect.right - outputMonitorRect.left;
	int outputMonitorHeight = outputMonitorRect.bottom - outputMonitorRect.top;

	// Set main window size/pos
	int mainWinX1 = outputMonitorRect.left;
	int mainWinY1 = outputMonitorRect.top;
	
	int mainWinX2 = mainWinX1 + outputMonitorWidth / 3 * 2;
	int mainWinY2 = mainWinY1 + outputMonitorHeight;

	//int mainWinX2 = outputMonitorRect.right;
	//int mainWinY2 = outputMonitorRect.bottom;

	RECT mainWinRect = { mainWinX1, mainWinY1, mainWinX2, mainWinY2 };
	cvNamedWindow(cMainWindowName.c_str(), CV_WINDOW_NORMAL);
	HWND mainWinHwnd = FindWindowA(0, cMainWindowName.c_str());
	SetWindow2Rect(mainWinHwnd, mainWinRect);
	//SetWindow2Rect(sideWinHwnd, mainWinRect);
	cv::imshow(cMainWindowName.c_str(), cv::Mat(1, 1, CV_8UC3, cv::Scalar(128, 128, 128)));
	cv::waitKey(33);

	// clean up monitor info
	delete(temp_vMonitorInfo);
}


/// <summary>
/// Initializes the second window
/// </summary>
/// <returns>indicates success or failure</returns>
void AnonymousKinect::InitSecondWindow()
{
	// Get monitor info
	vector<MONITORINFO> *temp_vMonitorInfo = new vector<MONITORINFO>;
	EnumDisplayMonitors(NULL, NULL, MonitorEnumProc, (LPARAM)temp_vMonitorInfo);
	RECT outputMonitorRect = temp_vMonitorInfo->at(cOutputMonitor).rcWork;
	int outputMonitorWidth = outputMonitorRect.right - outputMonitorRect.left;
	int outputMonitorHeight = outputMonitorRect.bottom - outputMonitorRect.top;

	// Set main window size/pos
	int mainWinX1 = outputMonitorRect.left + outputMonitorWidth / 3 * 2;
	int mainWinY1 = outputMonitorRect.top;
	int mainWinX2 = outputMonitorWidth;
	int mainWinY2 = outputMonitorHeight/2;
	RECT mainWinRect = { mainWinX1, mainWinY1, mainWinX2, mainWinY2 };
	cvNamedWindow(cSecondWindowName.c_str(), CV_WINDOW_NORMAL);
	HWND sideWinHwnd = FindWindowA(0, cSecondWindowName.c_str());
	SetWindow2Rect(sideWinHwnd, mainWinRect);
	cv::imshow(cSecondWindowName.c_str(), cv::Mat(1, 1, CV_8UC3, cv::Scalar(128, 128, 128)));
	cv::waitKey(33);

	// clean up monitor info
	delete(temp_vMonitorInfo);
}


/// <summary>
/// Enumerate Monitors Callback
/// </summary>
BOOL CALLBACK AnonymousKinect::MonitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData)
{
	vector<MONITORINFO>* vMonitorInfo = (vector<MONITORINFO>*) dwData;
	MONITORINFO mi;
	mi.cbSize = sizeof(mi);
	GetMonitorInfo(hMonitor, &mi);
	vMonitorInfo->push_back(mi);
	return TRUE;
}

/// <summary>
/// Moves and resizes window
/// </summary>
/// <returns>indicates success or failure</returns>
void AnonymousKinect::SetWindow2Rect(HWND winHandle, RECT rect)
{
	int winWidth = rect.right - rect.left + 1;
	int winHeight = rect.bottom - rect.top + 1;
	int winTopLeft_x = rect.left;
	int winTopLeft_y = rect.top;

	MoveWindow(winHandle, winTopLeft_x, winTopLeft_y, winWidth, winHeight, TRUE);

	int x = 0;
}


/// <summary>
/// Checks if a specific key has been pressed
/// </summary>
/// <param name="vkcode">keyboard key code</param>
bool AnonymousKinect::keyWasPressed(int vkcode)
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


void AnonymousKinect::wait(int seconds)
{
	clock_t endwait;
	endwait = clock() + seconds * CLOCKS_PER_SEC;
	while (clock() < endwait) {}
}
