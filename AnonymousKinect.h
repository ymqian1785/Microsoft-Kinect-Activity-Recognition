//------------------------------------------------------------------------------
// AnonymousKinect.h
// 
// University of Florida
// Francesco Pittaluga
// f.pittaluga@ufl.edu
//------------------------------------------------------------------------------

#pragma once
#include <Windows.h>

class AnonymousKinect
{


static const int        cDisplayMonitor = 1;
static const int		cOutputMonitor  = 0;


public:
	/// <summary>
	/// Constructor
	/// </summary>
	AnonymousKinect();

	/// <summary>
	/// Destructor
	/// </summary>
	~AnonymousKinect();

	/// <summary>
	/// Main application function
	/// </summary>
	void Run();

private:

	// 
	std::string				cDisplayCallibFileName;
	std::string				cInfraredCallibFileName;
	std::string				cConsoleName;
	std::string				cMainWindowName;
	std::string				cDisplayWindowName;
	std::string				cSecondWindowName;

	/// <summary>
	/// Initializes the console window
	/// </summary>
	/// <returns>indicates success or failure</returns>
	void					InitConsoleWindow();

	/// <summary>
	/// Create new display window
	/// </summary>
	/// <returns>indicates success or failure</returns>
	void					InitDisplayWindow();

	/// <summary>
	/// Creates new main output window
	/// </summary>
	/// <returns>indicates success or failure</returns>
	void					InitMainWindow();

	/// <summary>
	/// Initializes the second window
	/// </summary>
	/// <returns>indicates success or failure</returns>
	void					InitSecondWindow();

	/// <summary>
	/// Moves and resizes window
	/// </summary>
	/// <returns>indicates success or failure</returns>
	void					SetWindow2Rect(HWND winHandle, RECT rect);


	/// <summary>
	/// Enumerate Monitors Callback
	/// </summary>
	static BOOL CALLBACK	MonitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData);


	/// <summary>
	/// Checks if a specific key has been pressed
	/// </summary>
	/// <param name="vkcode">keyboard key code</param>
	bool					keyWasPressed(int vkcode);

	void					wait(int seconds);
};

